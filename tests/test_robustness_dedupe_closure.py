"""Behavior and no-reaccretion guards for robustness dedupe closure."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rlrmp.eval.robustness_diagnostics import (
    build_summary,
    evaluate_stabilization_row,
    run_feedback_robustness_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_summary_preserves_common_contract_and_schema_extensions(tmp_path: Path) -> None:
    paths = {"evaluation": tmp_path / "evaluation.json"}
    components = {
        "evaluation": {"schema_version": "evaluation.v1"},
        "feedback": {"schema_version": "feedback.v2"},
    }

    payload = build_summary(
        [{"run_id": "row-a"}],
        schema_version="summary.v3",
        issue="issue-a",
        scope="synthetic diagnostic",
        row_order=("row-a",),
        paths=paths,
        repo_relative=lambda path: path.name,
        components=components,
        component_schema_names=("evaluation", "feedback"),
        extensions={"baseline_row": "row-a", "interpretation": {"status": "stable"}},
        source_output_extensions={"summary_json": "summary.json"},
    )

    assert payload == {
        "schema_version": "summary.v3",
        "issue": "issue-a",
        "scope": "synthetic diagnostic",
        "row_order": ["row-a"],
        "rows": [{"run_id": "row-a"}],
        "baseline_row": "row-a",
        "interpretation": {"status": "stable"},
        "source_outputs": {
            "evaluation": "evaluation.json",
            "summary_json": "summary.json",
        },
        "component_schemas": {
            "evaluation": "evaluation.v1",
            "feedback": "feedback.v2",
        },
    }


def test_feedback_orchestration_preserves_materialization_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlrmp.eval import checkpoint_selection as gru_checkpoint_selection
    from rlrmp.analysis.pipelines import gru_feedback_ablation

    calls: list[tuple[str, dict[str, Any]]] = []
    checkpoint_path = tmp_path / "checkpoint.json"
    evaluation_path = tmp_path / "evaluation.json"
    detail_path = tmp_path / "detail.json"
    perturbation_path = tmp_path / "perturbation.json"
    checkpoint_path.touch()
    evaluation_path.touch()
    detail_path.touch()
    perturbation_path.touch()
    paths = {
        "checkpoint_manifest": checkpoint_path,
        "evaluation": evaluation_path,
        "evaluation_regeneration_spec": tmp_path / "evaluation-regeneration.json",
        "perturbation": perturbation_path,
        "perturbation_note": tmp_path / "perturbation.md",
        "perturbation_regeneration_spec": tmp_path / "perturbation-regeneration.json",
        "feedback": tmp_path / "feedback.json",
        "feedback_note": tmp_path / "feedback.md",
        "feedback_regeneration_spec": tmp_path / "feedback-regeneration.json",
    }

    def load_json(path: Path) -> dict[str, Any]:
        if path == checkpoint_path:
            return {"schema_version": "checkpoint.v1"}
        if path == evaluation_path:
            return {"schema_version": "evaluation.v1"}
        if path == perturbation_path:
            return {
                "schema_version": "perturbation.v1",
                "bulk_detail_manifest": {"path": str(detail_path)},
            }
        if path == detail_path:
            return {"detail": "loaded"}
        raise AssertionError(path)

    def feedback_materializer(**kwargs: Any) -> SimpleNamespace:
        calls.append(("feedback", kwargs))
        return SimpleNamespace(payload={"schema_version": "feedback.v1"})

    writes: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    hooks = {
        "load_json": load_json,
        "perturbation_output_is_current": lambda *_args, **_kwargs: True,
        "run_output_is_current": lambda *_args, **_kwargs: False,
    }
    monkeypatch.setattr(
        gru_checkpoint_selection,
        "build_validation_checkpoint_selection_manifest",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        gru_feedback_ablation,
        "execute_feedback_ablation_pipeline",
        feedback_materializer,
    )
    result = run_feedback_robustness_diagnostics(
        hooks=hooks,
        paths=paths,
        output_dirs=(tmp_path / "notes", tmp_path / "bulk"),
        issue="issue-a",
        repo_root=tmp_path,
        run_ids=("row-a",),
        labels=("Row A",),
        evaluation_bulk_dir=tmp_path / "evaluation-bulk",
        feedback_scope="feedback-scope",
        materialize_extensions=lambda _paths, _components: {
            "stabilization": {"schema_version": "stabilization.v1"}
        },
        build_rows=lambda components: [
            {
                "run_id": "row-a",
                "detail": components["perturbation_detail"]["detail"],
            }
        ],
        build_summary_payload=lambda rows, components: {
            "rows": list(rows),
            "component_count": len(components),
        },
        write_outputs=lambda summary, rows: writes.append((dict(summary), list(rows))),
    )

    assert result["rows"] == [{"run_id": "row-a", "detail": "loaded"}]
    assert result["summary"] == {"rows": result["rows"], "component_count": 6}
    assert writes == [(result["summary"], result["rows"])]
    assert [name for name, _kwargs in calls] == ["feedback"]
    feedback_kwargs = calls[0][1]
    assert feedback_kwargs["scope"] == "feedback-scope"
    assert feedback_kwargs["feedback_selection_level"] == "moderate"


def test_stabilization_evaluator_preserves_missing_family_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from feedbax.config import namespace

    from rlrmp.analysis.pipelines import cs_gru_standard_materialization
    from rlrmp.eval import checkpoint_selection as gru_checkpoint_selection
    from rlrmp.eval import perturbation_bank as gru_perturbation_bank
    from rlrmp.eval import trial_inputs
    from rlrmp.analysis.pipelines import gru_steady_state_perturbation_bank
    from rlrmp.eval import sisu_spectrum
    from rlrmp.train import task_model

    base = SimpleNamespace(command=np.zeros((2, 3, 1)), dt=0.01)
    run = SimpleNamespace(
        run_id="row-a",
        run_spec={"hps": {}, "seed": 7},
        run_spec_path=tmp_path / "run.json",
        artifact_dir=tmp_path / "artifacts",
    )
    hps = SimpleNamespace(model=SimpleNamespace(n_replicates=2))
    pair = SimpleNamespace(task=SimpleNamespace(validation_trials="validation"))
    probes = [
        SimpleNamespace(
            perturbation_id="command",
            group="mechanical",
            family="command_input_pulse",
            row="command",
        ),
        SimpleNamespace(
            perturbation_id="process",
            group="mechanical",
            family="process_epsilon_force_state_xy",
            row="process",
        ),
    ]

    class Adapter:
        def __init__(self, status: str):
            self.status = status
            self.reason = None if status == "evaluated" else "unsupported"
            self.model = None
            self.trial_specs = "steady"

        def to_json(self) -> dict[str, Any]:
            return {"status": self.status}

    hooks = {
        "build_probes": lambda **_kwargs: probes,
        "summarize_probe": lambda **_kwargs: {"family": "command_input_pulse"},
        "summarize_by_family": lambda _details: {
            "command_input_pulse": {"auc_displacement_mm_s_mean": 1.25}
        },
        "summarize_by_group": lambda _details: {
            "feedback": {
                "auc_displacement_mm_s_mean": 2.0,
                "peak_displacement_mm_mean": 3.0,
            },
            "mechanical": {
                "auc_displacement_mm_s_mean": 4.0,
                "peak_displacement_mm_mean": 5.0,
            },
        },
        "washin_diagnostics": lambda *_args, **_kwargs: {"status": "stable"},
        "repo_relative": lambda path, _root: path.name,
        "checkpoint_selection_summary": lambda selection: selection,
        "response_label": lambda _washin: "stable",
    }
    monkeypatch.setattr(namespace, "dict_to_namespace", lambda _payload, **_kwargs: hps)
    monkeypatch.setattr(
        cs_gru_standard_materialization,
        "normalize_gru_hps",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        trial_inputs,
        "resolve_evaluation_run_inputs",
        lambda **_kwargs: [run],
    )
    monkeypatch.setattr(
        trial_inputs,
        "repeat_single_validation_trial",
        lambda *_args: "repeated",
    )
    monkeypatch.setattr(
        gru_checkpoint_selection,
        "load_validation_selected_checkpoint_model",
        lambda **_kwargs: ("model", {"selection": "validation"}),
    )
    monkeypatch.setattr(task_model, "setup_task_model_pair", lambda *_args, **_kwargs: pair)
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "make_steady_state_trial_specs",
        lambda *_args, **_kwargs: (
            "steady",
            {"pulse_start_step": 5, "pulse_duration_steps": 3},
        ),
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "_target_position",
        lambda *_args: [0.1, 0.0],
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "pad_feedback_offset_inputs",
        lambda trials, **_kwargs: trials,
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "_expected_feedback_dim_from_hps",
        lambda _hps: 6,
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "_feedback_dim",
        lambda _trials: 6,
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "_evaluate_model_on_trial_specs",
        lambda **_kwargs: base,
    )
    monkeypatch.setattr(
        gru_steady_state_perturbation_bank,
        "washin_diagnostics",
        lambda *_args, **_kwargs: {"status": "stable"},
    )
    monkeypatch.setattr(
        sisu_spectrum,
        "zero_disturbance_payload",
        lambda trials: trials,
    )
    monkeypatch.setattr(
        gru_perturbation_bank,
        "apply_perturbation_to_trial_specs",
        lambda _trials, row, **_kwargs: Adapter(
            "evaluated" if row == "command" else "not_applicable"
        ),
    )
    row = SimpleNamespace(run_id="row-a", training="pgd", physical_level="moderate")

    result = evaluate_stabilization_row(
        row,
        repo_root=tmp_path,
        hooks=hooks,
        source_experiment="issue-a",
        row_metadata=lambda current: {
            "run_id": current.run_id,
            "training": current.training,
            "physical_level": current.physical_level,
        },
        allowed_missing_families=("process_epsilon_force_state_xy",),
    )

    assert result["command_input_auc_mm_s"] == 1.25
    assert result["process_force_auc_mm_s"] is None
    assert result["missing_families"] == ["process_epsilon_force_state_xy"]
    assert result["feedback_auc_mm_s"] == 2.0
    assert result["mechanical_peak_mm"] == 5.0
    assert result["run_spec_path"] == "run.json"
    assert result["checkpoint_selection_summary"] == {"selection": "validation"}

    with pytest.raises(KeyError, match="process_epsilon_force_state_xy"):
        evaluate_stabilization_row(
            row,
            repo_root=tmp_path,
            hooks=hooks,
            source_experiment="issue-a",
            row_metadata=lambda current: {"run_id": current.run_id},
        )


@pytest.mark.parametrize(
    ("relpath", "function_name", "canonical_call", "max_lines"),
    (
        (
            "results/b413bb0/scripts/materialize_beta1p4_stabilization_diagnostics.py",
            "evaluate_row_allowing_missing_families",
            "canonical_evaluate_stabilization_row",
            25,
        ),
        (
            "results/c92ebd8/scripts/materialize_pgd_robustness_isolation.py",
            "evaluate_stabilization_row",
            "canonical_evaluate_stabilization_row",
            30,
        ),
        (
            "results/c92ebd8/scripts/materialize_pgd_1p05_stabilization_diagnostics.py",
            "evaluate_row",
            "canonical_evaluate_stabilization_row",
            20,
        ),
        (
            "results/d55c5f0/scripts/materialize_soft_pgd_feedback_robustness_diagnostics.py",
            "main",
            "run_feedback_robustness_diagnostics",
            50,
        ),
        (
            "results/c92ebd8/scripts/materialize_ofb_budget_feedback_robustness_diagnostics.py",
            "main",
            "run_feedback_robustness_diagnostics",
            50,
        ),
        (
            "results/c92ebd8/scripts/materialize_pgd_1p05_reach_context_diagnostics.py",
            "main",
            "run_feedback_robustness_diagnostics",
            50,
        ),
        (
            "results/d55c5f0/scripts/materialize_soft_pgd_feedback_robustness_diagnostics.py",
            "build_summary",
            "canonical_build_summary",
            50,
        ),
        (
            "results/c92ebd8/scripts/materialize_ofb_budget_feedback_robustness_diagnostics.py",
            "build_summary",
            "canonical_build_summary",
            50,
        ),
        (
            "results/c92ebd8/scripts/materialize_pgd_1p05_reach_context_diagnostics.py",
            "build_summary",
            "canonical_build_summary",
            50,
        ),
    ),
)
def test_manifest_member_is_a_thin_canonical_adapter(
    relpath: str,
    function_name: str,
    canonical_call: str,
    max_lines: int,
) -> None:
    tree = ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert canonical_call in calls
    assert function.end_lineno is not None
    assert function.end_lineno - function.lineno + 1 <= max_lines
    if function_name in {
        "evaluate_row_allowing_missing_families",
        "evaluate_stabilization_row",
        "evaluate_row",
    }:
        assert not any(isinstance(node, (ast.For, ast.While, ast.Try)) for node in ast.walk(function))
    if function_name == "main":
        assert calls.isdisjoint(
            {
                "build_validation_checkpoint_selection_manifest",
                "evaluate_gru_diagnostics_runs",
                "materialize_gru_perturbation_response",
                "materialize_gru_feedback_ablation",
            }
        )

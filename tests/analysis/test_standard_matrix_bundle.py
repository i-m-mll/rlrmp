"""Tests for the declarative standard-matrix analysis bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

import rlrmp
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    execute_staged_analysis_bundle,
    load_analysis_bundle,
)
from feedbax.analysis.evaluation import (
    execute_evaluation_run_spec,
)
from feedbax.analysis.specs import execute_analysis_run_spec
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    FigureManifest,
    ParentRef,
    load_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from rlrmp.analysis.matrix import STANDARD_MATRIX_ANALYSIS_TYPE, STANDARD_MATRIX_EVALUATION_TYPE
from rlrmp.analysis.matrix.standard_matrix import _notes_path
from rlrmp.paths import REPO_ROOT


def _registry() -> ExperimentRegistry:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    return registry


def _load_standard_matrix_bundle(registry: ExperimentRegistry) -> AnalysisBundleSpec:
    errors: list[str] = []
    for key in ("rlrmp/standard_matrix", "rlrmp/standard-matrix"):
        try:
            return load_analysis_bundle(key, registry=registry)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise AssertionError(
        "Expected rlrmp's standard-matrix analysis bundle to be loadable as "
        "'rlrmp/standard_matrix' or 'rlrmp/standard-matrix'. Loader errors: " + " | ".join(errors)
    )


def test_standard_matrix_bundle_loads_expands_and_executes_lightweight_routed_path(
) -> None:
    registry = _registry()
    bundle = _load_standard_matrix_bundle(registry)
    assert not bundle.templates
    assert bundle.stages, "standard-matrix bundle should declare staged figure execution"
    assert bundle.metadata.get("figure_routing", {}).get("package") == "rlrmp"
    assert [stage.name for stage in bundle.stages[:3]] == [
        "figure_payload",
        "forward_velocity_profiles",
        "hold_drift_profiles",
    ]
    assert bundle.stages[0].kind == "analysis"
    assert bundle.stages[1].kind == "figure"
    assert bundle.stages[1].figure.template == "rlrmp.profile_comparison"
    assert bundle.stages[1].depends_on == ["figure_payload"]


def test_registered_standard_matrix_recipe_executes_profile_with_routing(
    tmp_path: Path,
) -> None:
    registry = _registry()
    payload = {
        "cells": [
            {
                "run_id": "cell-a",
                "label": "cell_a",
                "display_name": "Cell A",
                "color": "#1f77b4",
                "forward_velocity": {
                    "time": [0, 1, 2],
                    "mean": [0.0, 0.6, 0.2],
                    "lower": [-0.1, 0.5, 0.1],
                    "upper": [0.1, 0.7, 0.3],
                },
                "summary_metrics": {
                    "velocity_rmse": 0.12,
                    "peak_velocity": 0.6,
                },
            },
            {
                "run_id": "cell-b",
                "label": "cell_b",
                "display_name": "Cell B",
                "color": "#ff7f0e",
                "forward_velocity": {
                    "time": [0, 1, 2],
                    "mean": [0.0, 0.4, 0.1],
                    "lower": [-0.1, 0.3, 0.0],
                    "upper": [0.1, 0.5, 0.2],
                },
                "summary_metrics": {
                    "velocity_rmse": 0.18,
                    "peak_velocity": 0.4,
                },
            },
        ]
    }
    eval_spec = EvaluationRunSpec(
        evaluation_type=STANDARD_MATRIX_EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="rlrmp-test-training-run:real-standard-matrix",
                role="training_run",
            )
        ],
        params={"matrix_payload": payload, "legacy_payload_mode": True},
    )
    eval_manifest, _eval_path = execute_evaluation_run_spec(
        eval_spec,
        root=tmp_path,
        metadata={"source": "test"},
        issues=["63cec06"],
        force=True,
    )

    bundle = _load_standard_matrix_bundle(registry)
    execution = execute_staged_analysis_bundle(
        bundle,
        root=tmp_path,
        run_ids=[eval_manifest.id],
        issues=["63cec06"],
    )

    forward_stage = next(
        stage for stage in execution.stages if stage.name == "forward_velocity_profiles"
    )
    assert forward_stage.status == "materialized"
    figure_ref = forward_stage.manifest_refs[0]
    assert figure_ref.kind == "FigureManifest"
    figure_manifest = load_manifest(Path(figure_ref.uri))
    assert isinstance(figure_manifest, FigureManifest)
    assert figure_manifest.status == "completed"
    assert figure_manifest.figure_spec.inline["template"] == "rlrmp.profile_comparison"
    assert figure_manifest.figure_spec.inline["metadata"]["profile_key"] == "forward_velocity"
    assert figure_manifest.provenance.issues == ["63cec06"]
    render_artifacts = [
        artifact for artifact in figure_manifest.artifacts if artifact.role == "figure_render"
    ]
    assert render_artifacts
    assert any(Path(artifact.uri).exists() for artifact in render_artifacts if artifact.uri)


def test_registered_standard_matrix_notes_preserve_handwritten_sections(
    tmp_path: Path,
) -> None:
    _registry()
    payload = {
        "cells": [
            {
                "run_id": "cell-a",
                "label": "cell_a",
                "display_name": "Cell A",
                "summary_metrics": {
                    "velocity_rmse": 0.12,
                    "peak_velocity": 0.6,
                },
            }
        ]
    }
    eval_spec = EvaluationRunSpec(
        evaluation_type=STANDARD_MATRIX_EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="rlrmp-test-training-run:standard-matrix-notes",
                role="training_run",
            )
        ],
        params={"matrix_payload": payload, "legacy_payload_mode": True},
    )
    eval_manifest, eval_path = execute_evaluation_run_spec(
        eval_spec,
        root=tmp_path,
        issues=["63cec06"],
        force=True,
    )
    notes_path = tmp_path / "matrix_results.md"
    notes_path.write_text("# Handwritten preamble\n\nKeep this.\n", encoding="utf-8")

    analysis_spec = AnalysisRunSpec(
        analysis_type=STANDARD_MATRIX_ANALYSIS_TYPE,
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id=eval_manifest.id,
                role="evaluation_run",
                uri=str(eval_path),
            )
        ],
        params={
            "requested_outputs": ["notes"],
            "metric_order": ["velocity_rmse", "peak_velocity"],
            "note_marker": "standard_matrix",
            "notes_path": str(notes_path),
        },
    )
    manifest, _manifest_path = execute_analysis_run_spec(
        analysis_spec,
        root=tmp_path,
        issues=["63cec06"],
        fig_dump_formats=("json",),
    )

    text = notes_path.read_text(encoding="utf-8")
    assert text.startswith("# Handwritten preamble\n\nKeep this.\n")
    assert "<!-- AUTO-GENERATED: standard_matrix -->" in text
    assert "| Cell | velocity_rmse | peak_velocity |" in text
    assert "| Cell A | 0.12 | 0.6 |" in text
    assert manifest.summary_metrics["analysis_count"] == 1
    assert manifest.summary_metrics["artifact_count"] == 1
    assert manifest.artifacts[0].role == "analysis_notes"


def test_standard_matrix_default_notes_path_is_repo_root_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _notes_path(
        None,  # type: ignore[arg-type]
        {"figure_routing": {"experiment": "5c302e2"}},
    ) == REPO_ROOT / "results" / "5c302e2" / "notes" / "matrix_results.md"

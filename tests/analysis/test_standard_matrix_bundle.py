"""Tests for the declarative standard-matrix analysis bundle."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import plotly.graph_objects as go
import pytest

import rlrmp
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    execute_analysis_bundle,
    expand_analysis_bundle,
    load_analysis_bundle,
    select_bundle_manifests,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.contracts.manifest import AnalysisRunSpec, EvaluationRunSpec, ParentRef, load_manifest
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.analysis.types import AnalysisInputData
from feedbax.config.namespace import TreeNamespace
from rlrmp.analysis.matrix import STANDARD_MATRIX_ANALYSIS_TYPE, STANDARD_MATRIX_EVALUATION_TYPE
from rlrmp.analysis.matrix.standard_matrix import _notes_path
from rlrmp.paths import REPO_ROOT


TOY_EVALUATION_TYPE = "rlrmp_test.standard_matrix_eval"


class TinyMatrixAnalysis(AbstractAnalysis):
    """Small analysis payload used to exercise bundle execution and routing."""

    def compute(self, data: AnalysisInputData, **_kwargs):
        return {
            "value": int(data.states["value"]) + 1,
            "params": dict(data.states.get("params", {})),
        }

    def make_figs(self, data: AnalysisInputData, *, result, **_kwargs):
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[int(data.states["value"]), result["value"]])
        return {"main": fig}

    def _params_to_save(self, hps, *, result, **_kwargs):
        return {
            "result_value": result["value"],
            **result.get("params", {}),
        }


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


def _assign_dotted(target: dict[str, object], key: str, value: object) -> None:
    parts = key.split(".")
    current = target
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot assign dotted key {key!r} through scalar {part!r}")
        current = next_value
    current[parts[-1]] = value


def _nested_payload(equals: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in equals.items():
        _assign_dotted(payload, key, value)
    return payload


def _register_toy_evaluation_recipe() -> None:
    def recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path):
        return EvaluationRecipeResult(
            states={"value": run_spec.params.get("n_trials", 1)},
            summary_metrics={"n_trials": run_spec.params.get("n_trials", 1)},
        )

    register_evaluation_recipe(TOY_EVALUATION_TYPE, recipe, replace=True)


def _write_matching_evaluation_manifest(
    root: Path,
    bundle: AnalysisBundleSpec,
) -> tuple[str, Path]:
    if bundle.predicate.manifest_kind != "EvaluationRunManifest":
        raise AssertionError(
            "The standard-matrix bundle should select EvaluationRunManifest inputs, "
            f"not {bundle.predicate.manifest_kind!r}."
        )
    if bundle.predicate.run_ids:
        raise AssertionError(
            "The standard-matrix bundle should not hard-code run IDs; tests need to "
            "apply it to fresh manifest roots."
        )

    parent = ParentRef(
        kind="TrainingRunManifest",
        id="rlrmp-test-training-run:standard-matrix",
        role="training_run",
    )
    params = {"n_trials": 1, **_nested_payload(bundle.predicate.params_equals)}
    spec = EvaluationRunSpec(
        evaluation_type=TOY_EVALUATION_TYPE,
        inputs=[parent],
        params=params,
    )
    manifest, path = execute_evaluation_run_spec(
        spec,
        root=root,
        metadata=_nested_payload(bundle.predicate.metadata_equals),
        issues=["63cec06"],
        force=True,
    )
    return manifest.id, path


def _register_toy_analysis_recipes(analysis_types: Iterable[str]) -> None:
    def recipe(spec, _root: Path, inputs):
        outputs = spec.params.get("requested_outputs", spec.params.get("outputs")) or ["toy"]
        analyses = {
            str(output): TinyMatrixAnalysis(variant=str(output), cache_result=True)
            for output in outputs
        }
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses=analyses,
            data=AnalysisInputData(
                models={},
                tasks={},
                states={"value": value, "params": dict(spec.params)},
                hps={"toy": TreeNamespace(task=TreeNamespace(eval_n=1))},
                extras={},
            ),
        )

    for analysis_type in analysis_types:
        register_analysis_recipe(analysis_type, recipe, replace=True)


def _unregister_toy_analysis_recipes(analysis_types: Iterable[str]) -> None:
    for analysis_type in analysis_types:
        unregister_analysis_recipe(analysis_type)


def test_standard_matrix_bundle_loads_expands_and_executes_lightweight_routed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    bundle = _load_standard_matrix_bundle(registry)
    assert bundle.templates, "standard-matrix bundle should declare executable templates"
    assert bundle.metadata.get("figure_routing", {}).get("package") == "rlrmp"

    _register_toy_evaluation_recipe()
    analysis_types = {template.analysis_type for template in bundle.templates}
    _register_toy_analysis_recipes(analysis_types)
    try:
        manifest_id, _manifest_path = _write_matching_evaluation_manifest(tmp_path, bundle)
        matched = select_bundle_manifests(bundle, tmp_path, run_ids=[manifest_id])
        assert [manifest.id for manifest in matched] == [manifest_id]

        expansions = expand_analysis_bundle(bundle, matched)
        assert expansions
        assert all(expansion.matched_run_ids == (manifest_id,) for expansion in expansions)
        assert {expansion.spec.analysis_type for expansion in expansions} <= analysis_types

        import feedbax.plugins as plugins
        import feedbax.plot.io as plot_io

        fake_repo_root = tmp_path / "routed_repo"
        monkeypatch.setattr(plugins, "EXPERIMENT_REGISTRY", registry)
        monkeypatch.setattr(plot_io, "_find_repo_root", lambda _module: fake_repo_root)

        outputs = execute_analysis_bundle(
            bundle,
            root=tmp_path,
            run_ids=[manifest_id],
            issues=["63cec06"],
            fig_dump_formats=("json",),
        )

        assert outputs
        for expansion, manifest, manifest_path in outputs:
            assert manifest_path.exists()
            assert manifest.status == "completed"
            assert manifest.provenance.issues == ["63cec06"]
            assert manifest.metadata["bundle"]["name"] == bundle.name
            assert manifest.metadata["bundle"]["template"] == expansion.template_name
            assert manifest.metadata["bundle"]["matched_run_ids"] == [manifest_id]
            assert manifest.summary_metrics["figure_count"] >= 1

            loaded = load_manifest(manifest_path)
            assert loaded.id == manifest.id
            routed = [
                artifact.metadata["figure_routing"]
                for artifact in manifest.artifacts
                if artifact.role == "figure" and "figure_routing" in artifact.metadata
            ]
            assert routed, "fake figure output should be projected through rlrmp routing"
            for projection in routed:
                spec_path = Path(projection["spec_path"])
                render_path = Path(projection["render_path"])
                symlink_path = Path(projection["symlink_path"])
                assert spec_path.is_relative_to(fake_repo_root)
                assert render_path.is_relative_to(fake_repo_root)
                assert spec_path.exists()
                assert render_path.exists()
                assert symlink_path.is_symlink()
                routed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                assert routed_spec["analysis"]["manifest_id"] == manifest.id
                assert routed_spec["analysis"]["analysis_type"] == expansion.spec.analysis_type
    finally:
        _unregister_toy_analysis_recipes(analysis_types)
        rlrmp.register_experiment_package(registry)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_registered_standard_matrix_recipe_executes_profile_with_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    eval_manifest, eval_path = execute_evaluation_run_spec(
        eval_spec,
        root=tmp_path,
        metadata={"source": "test"},
        issues=["63cec06"],
        force=True,
    )

    import feedbax.plugins as plugins
    import feedbax.plot.io as plot_io

    fake_repo_root = tmp_path / "routed_repo"
    monkeypatch.setattr(plugins, "EXPERIMENT_REGISTRY", registry)
    monkeypatch.setattr(plot_io, "_find_repo_root", lambda _module: fake_repo_root)

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
            "requested_outputs": ["forward_velocity_profiles"],
            "profile_key": "forward_velocity",
            "figure_routing": {
                "package": "rlrmp",
                "experiment": "63cec06",
                "topic": "forward_velocity_profiles",
                "extra_packages": ["rlrmp"],
            },
        },
    )
    manifest, manifest_path = execute_analysis_run_spec(
        analysis_spec,
        root=tmp_path,
        issues=["63cec06"],
        fig_dump_formats=("json",),
    )

    assert manifest_path.exists()
    assert manifest.status == "completed"
    assert manifest.summary_metrics["analysis_count"] == 1
    assert manifest.summary_metrics["figure_count"] == 1
    assert manifest.provenance.issues == ["63cec06"]
    assert manifest.inputs[0].id == eval_manifest.id
    assert load_manifest(manifest_path).id == manifest.id

    figure_artifact = next(artifact for artifact in manifest.artifacts if artifact.role == "figure")
    projection = figure_artifact.metadata["figure_routing"]
    spec_path = Path(projection["spec_path"])
    render_path = Path(projection["render_path"])
    assert spec_path == (
        fake_repo_root
        / "results"
        / "63cec06"
        / "figures"
        / "forward_velocity_profiles"
        / "spec.json"
    )
    assert render_path.exists()
    routed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert routed_spec["analysis"]["analysis_type"] == STANDARD_MATRIX_ANALYSIS_TYPE
    assert routed_spec["plot_kwargs"]["params"]["standard_matrix_output"] == (
        "forward_velocity_profiles"
    )


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

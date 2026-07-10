from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from rlrmp.viz.figures import (
    build_nominal_profile_figure,
    build_nominal_velocity_spec,
    build_profile_family_figure,
    build_stabilization_family_figure,
)
from rlrmp.viz.traces import add_band_trace, add_profile_line


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_trace_helpers_preserve_band_and_profile_contracts() -> None:
    fig = go.Figure()
    add_band_trace(
        fig,
        x=np.array([0.0, 1.0]),
        mean=np.array([1.0, 2.0]),
        std=np.array([0.1, 0.2]),
        color="#336699",
        name="band",
    )
    add_profile_line(
        fig,
        np.array([[1.0, 3.0], [3.0, 5.0]]),
        row=None,
        col=None,
        name="mean",
        color="#000000",
        dash="solid",
        showlegend=True,
    )

    assert len(fig.data) == 3
    np.testing.assert_allclose(np.asarray(fig.data[-1].y), [2.0, 4.0])
    np.testing.assert_allclose(np.asarray(fig.data[-1].x), [0.0, 0.01])


def test_nominal_figure_and_spec_helpers_keep_shared_axis_contract() -> None:
    rows = [{"label": "row a", "run_id": "a"}, {"label": "row b", "run_id": "b"}]

    def add_run(fig: go.Figure, _row: dict[str, str], row_index: int) -> None:
        add_profile_line(
            fig,
            np.array([0.0, 1.0]),
            row=row_index,
            col=1,
            name=f"run {row_index}",
            color="#2563eb",
            dash="solid",
            showlegend=True,
        )

    fig = build_nominal_profile_figure(
        rows=rows,
        add_run_profiles=add_run,
        ext_profile=np.array([0.0, 0.5]),
        robust_profile=np.array([0.0, 0.4]),
        comparator_colors={"extlqg6d": "#111827", "robust_output_feedback6d": "#dc2626"},
        title="comparison",
        height=600,
    )
    spec = build_nominal_velocity_spec(
        schema_version="test.v1",
        issue="test",
        figure_kind="test",
        robust_contract={"label": "robust"},
        inputs=[],
        transform_name="profiles",
        transform_kwargs={},
        rows=2,
        outputs={"html": "figure.html"},
    )

    assert len(fig.data) == 6
    assert fig.layout.yaxis.matches is not None
    assert spec["plot_kwargs"]["shared_yaxes"] == "all"


def test_profile_family_builder_preserves_timing_grid_and_event_marker() -> None:
    rows = [{"timing": "early"}, {"timing": "late"}]

    def add_trace(fig, samples, *, row, col, showlegend, **_kwargs):
        fig.add_trace(
            go.Scatter(y=np.mean(samples, axis=0), showlegend=showlegend),
            row=row,
            col=col,
        )

    fig = build_profile_family_figure(
        rows,
        quantity_specs=(("velocity", "Velocity", "m/s"),),
        timing_bins_for_rows=lambda _rows: ("early", "late"),
        row_timing_label=lambda row: row["timing"],
        representative_timing=lambda _rows: {"start": 1.0, "stop": 2.0},
        perturbation_interval_bounds=lambda timing: (timing["start"], timing["stop"]),
        collect_traces=lambda _rows: {
            ("gru", "residual", "velocity", "orthogonal"): np.ones((2, 3))
        },
        trace_key=lambda source, variant, quantity, coord: (
            source,
            variant,
            quantity,
            coord,
        ),
        add_trace=add_trace,
        sources=("gru",),
        variants=("residual",),
        axis_unit=lambda _quantity, unit: unit,
        figure_kind="residual",
        title="profiles",
        width_min=800,
        width_per_column=300,
        height=500,
    )

    assert len(fig.data) == 2
    assert len(fig.layout.shapes) == 2
    assert fig.layout.yaxis2.matches == "y"
    assert fig.layout.xaxis2.title.text == "time from movement onset (s)"


def test_stabilization_builder_collects_cell_metadata() -> None:
    def render_cell(fig, response, column, row, col, legend_seen, outputs):
        del legend_seen
        fig.add_trace(go.Scatter(y=[row, col]), row=row, col=col)
        outputs["coverage"].append({"response": response, "column": column})
        outputs["event_markers"].append({"row": row, "col": col})

    fig, coverage, markers, unavailable = build_stabilization_family_figure(
        response_variables=("position", "velocity"),
        columns=("small", "large"),
        response_label=str.title,
        column_label=str.title,
        response_axis_title=lambda response: f"{response} units",
        render_cell=render_cell,
        title="stabilization",
        width=900,
        horizontal_spacing=0.05,
    )

    assert len(fig.data) == 4
    assert len(coverage) == 4
    assert len(markers) == 4
    assert unavailable == []
    assert fig.layout.yaxis.title.text == "position units"


def test_owned_scalar_helpers_are_import_routed_not_redefined() -> None:
    expected_absent = {
        "results/40e1911/scripts/materialize_delayed_timing_hold_lane_velocity_profiles.py": {
            "add_band_trace",
            "add_reference_trace",
            "rgba",
            "initial_effector_position",
            "initial_effector_velocity",
        },
        "results/6c36536/scripts/materialize_delayed_movement_bank_velocity_profiles.py": {
            "add_band_trace",
            "add_reference_trace",
            "rgba",
            "initial_effector_position",
            "initial_effector_velocity",
        },
        "results/91a090c/scripts/materialize_nominal_velocity_profiles.py": {
            "initial_effector_velocity",
            "json_ready",
            "materialize_analytical_profiles",
            "repo_ref",
        },
        "results/e148f33/scripts/materialize_nominal_velocity_profile_comparison.py": {
            "initial_effector_velocity",
            "json_ready",
            "materialize_analytical_profiles",
            "repo_ref",
            "rgba",
        },
    }
    for relative_path, names in expected_absent.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert definitions.isdisjoint(names), (relative_path, definitions & names)


def test_visual_figure_cluster_members_route_through_canonical_builders() -> None:
    expected_calls = {
        "results/3244f1a/scripts/materialize_final_small_bank_profiles.py": {
            "build_profile_family_figure"
        },
        "results/c92ebd8/scripts/materialize_pgd_1p05_perturbation_response_overlays.py": {
            "build_profile_family_figure"
        },
        "results/c92ebd8/scripts/materialize_pgd_1p05_moderate_perturbation_profiles_overlay.py": {
            "build_profile_family_figure"
        },
        "results/c92ebd8/scripts/materialize_post_training_figures.py": {
            "build_profile_family_figure",
            "build_nominal_profile_figure",
        },
        "results/c92ebd8/scripts/materialize_pgd_1p05_nominal_velocity_profiles.py": {
            "build_nominal_profile_figure"
        },
        "results/c92ebd8/scripts/materialize_pgd_ofb_budget_stabilization_responses.py": {
            "build_stabilization_family_figure"
        },
        "results/d55c5f0/scripts/materialize_soft_pgd_stabilization_responses.py": {
            "build_stabilization_family_figure"
        },
    }
    for relative_path, expected in expected_calls.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert expected <= calls, (relative_path, expected - calls)
        assert "make_subplots" not in calls, relative_path
        assert "profile_comparison_grid" not in calls, relative_path


def test_delayed_eval_cluster_members_are_thin_canonical_adapters() -> None:
    for relative_path in (
        "results/40e1911/scripts/materialize_delayed_timing_hold_lane_velocity_profiles.py",
        "results/6c36536/scripts/materialize_delayed_movement_bank_velocity_profiles.py",
    ):
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, canonical_name in (
            ("evaluate_velocity_profile", "canonical_evaluate_velocity_profile"),
            ("make_delayed_eval_bank", "canonical_make_delayed_eval_bank"),
        ):
            function = definitions[name]
            calls = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert canonical_name in calls, (relative_path, name)
            assert function.end_lineno - function.lineno < 30, (relative_path, name)

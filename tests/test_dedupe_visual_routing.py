from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from rlrmp.viz.figures import build_nominal_profile_figure, build_nominal_velocity_spec
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

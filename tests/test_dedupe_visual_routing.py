from __future__ import annotations

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

import rlrmp.viz.figures as figure_helpers
from rlrmp.viz.figures import (
    build_nominal_profile_figure,
    build_nominal_velocity_spec,
    build_profile_family_figure,
    build_stabilization_family_figure,
    build_stabilization_response_family_figure,
    materialize_nominal_velocity_figure,
    write_velocity_by_replicate_figure,
    write_velocity_figure,
)
from rlrmp.viz.traces import add_band_trace, add_profile_line


REPO_ROOT = Path(__file__).resolve().parents[1]


def _velocity_profile(label: str, *, bank_kind: str = "no_catch") -> SimpleNamespace:
    return SimpleNamespace(
        experiment="experiment",
        run_id=label,
        label=label,
        bank_kind=bank_kind,
        time_s=np.array([-0.01, 0.0, 0.01]),
        mean=np.array([0.0, 0.1, 0.2]),
        std=np.array([0.01, 0.02, 0.03]),
        replicate_mean=np.array([[0.0, 0.1, 0.2], [0.0, 0.2, 0.4]]),
        replicate_std=np.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]]),
        n_replicates=2,
        sisu_level=None,
    )


def _velocity_reference() -> SimpleNamespace:
    return SimpleNamespace(
        time_s=np.array([-0.01, 0.0, 0.01]),
        forward_velocity=np.array([0.0, 0.08, 0.16]),
        forward_velocity_std=np.array([0.01, 0.01, 0.02]),
        line_color="#111827",
        label="reference",
        observation_channel="position",
        line_dash="dash",
    )


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


def test_full_stabilization_family_builder_preserves_grid_and_sidecar_contracts() -> None:
    analytical_calls = []

    def cell_context(response, column):
        profile = {"response": response, "column": column}
        return {
            "timing": {"start": 1},
            "dt": 0.01,
            "baseline_rows": [profile],
            "learned": [{"source": "gru", "run_id": "run", "profile": profile}],
            "cache_prefix": (),
            "coverage_metadata": {"budget": column},
            "identity_metadata": {"budget_column": column},
        }

    def analytical_profile(**kwargs):
        analytical_calls.append(kwargs)
        if kwargs["source"] == "robust_output_feedback6d":
            return {"status": "unsupported", "reason": "unsupported"}
        return {"status": "available", "profile": {"source": kwargs["source"]}}

    def add_profile_traces(*, fig, source, row, col, **_kwargs):
        fig.add_trace(go.Scatter(y=[row, col], name=source), row=row, col=col)

    fig, coverage, markers, unavailable = build_stabilization_response_family_figure(
        family="command_input",
        response_variables=("position", "velocity", "command"),
        columns=("small", "large"),
        cell_context=cell_context,
        analytical_sources=("extlqg6d", "robust_output_feedback6d"),
        analytical_profile=analytical_profile,
        add_profile_traces=add_profile_traces,
        coverage_row=lambda **kwargs: kwargs,
        add_unsupported_annotation=lambda **kwargs: kwargs["fig"].add_annotation(
            text=kwargs["text"], row=kwargs["row"], col=kwargs["col"]
        ),
        infer_event_marker=lambda **_kwargs: {"time_s": 0.1},
        add_event_marker=lambda **kwargs: kwargs["fig"].add_vline(
            x=kwargs["marker"]["time_s"], row=kwargs["row"], col=kwargs["col"]
        ),
        response_label=str.title,
        column_label=str.title,
        response_axis_title=lambda response: f"{response} units",
        title="family response",
        width=1180,
        horizontal_spacing=0.07,
    )

    assert len(fig.data) == 12
    assert (fig.layout.width, fig.layout.height) == (1180, 900)
    assert [annotation.text for annotation in fig.layout.annotations[:6]] == [
        "Position - Small",
        "Position - Large",
        "Velocity - Small",
        "Velocity - Large",
        "Command - Small",
        "Command - Large",
    ]
    assert len(coverage) == 12
    assert coverage[0]["budget"] == "small"
    assert len(markers) == 6
    assert markers[0]["budget_column"] == "small"
    assert len(unavailable) == 6
    assert unavailable[0]["reason"] == "unsupported"
    assert len(analytical_calls) == 6


def test_nominal_materializer_preserves_spec_save_and_file_contracts(monkeypatch) -> None:
    saved = []

    def fake_save_figure(**kwargs):
        saved.append(kwargs)
        return {"artifact": "figure"}

    monkeypatch.setattr(figure_helpers, "save_figure", fake_save_figure)

    def add_run(fig, _row, row_index):
        add_profile_line(
            fig,
            np.array([0.0, 0.2]),
            row=row_index,
            col=1,
            name="GRU nominal",
            color="#2563eb",
            dash="solid",
            showlegend=row_index == 1,
        )

    config = {
        "title": "nominal profiles",
        "height": 760,
        "issue": "c92ebd8",
        "topic": "nominal",
        "schema_version": "nominal.v1",
        "figure_kind": "nominal_velocity",
        "ext_contract": {"label": "6D extLQG"},
        "inputs": [{"path": "results/c92ebd8/runs/a.json"}],
        "transform": [{"name": "forward_velocity"}],
        "plot_kwargs": {"shared_yaxes": "all", "rows": 2},
    }
    result = materialize_nominal_velocity_figure(
        rows=({"label": "a"}, {"label": "b"}),
        add_run_profiles=add_run,
        ext_profile=np.array([0.0, 0.1]),
        robust_profile=np.array([0.0, 0.08]),
        comparator_colors={"extlqg6d": "#111827", "robust_output_feedback6d": "#15803d"},
        config=config,
        robust_contract={"label": "robust"},
        result_extra={"bulk_html": "_artifacts/c92ebd8/figures/nominal/figure.html"},
        result_contract={"output_feedback_hinf": {"label": "robust"}},
    )

    assert len(saved) == 1
    assert len(saved[0]["fig"].data) == 6
    assert saved[0]["fig"].layout.title.text == "nominal profiles"
    assert saved[0]["spec"] == {
        "schema_version": "nominal.v1",
        "issue": "c92ebd8",
        "figure_kind": "nominal_velocity",
        "analytical_comparator_contract": {
            "extlqg": {"label": "6D extLQG"},
            "output_feedback_hinf": {"label": "robust"},
        },
        "inputs": [{"path": "results/c92ebd8/runs/a.json"}],
        "transform": [{"name": "forward_velocity"}],
        "plot_kwargs": {"shared_yaxes": "all", "rows": 2},
    }
    assert result["spec"] == "results/c92ebd8/figures/nominal/spec.json"
    assert result["html"] == "results/c92ebd8/figures/nominal/figure.html"
    assert result["bulk_html"] == "_artifacts/c92ebd8/figures/nominal/figure.html"
    assert result["analytical_comparator_contract"] == {
        "output_feedback_hinf": {"label": "robust"}
    }


def test_velocity_writer_preserves_single_and_sequence_output_contracts(
    tmp_path, monkeypatch
) -> None:
    writes: list[tuple[go.Figure, Path]] = []
    monkeypatch.setattr(
        go.Figure,
        "write_html",
        lambda figure, path: writes.append((figure, Path(path))),
    )
    first = _velocity_profile("first")
    second = _velocity_profile("second")
    reference = _velocity_reference()

    single_path = write_velocity_figure(
        first,
        output_dir=tmp_path,
        references=(reference,),
        title="Delayed movement-bank target-radial velocity ({bank_kind})",
    )
    single, single_written = writes.pop()
    assert single_path == single_written == tmp_path / "forward_velocity_profiles_stochastic.html"
    assert single.layout.title.text == "Delayed movement-bank target-radial velocity (no_catch)"
    assert (single.layout.width, single.layout.height) == (900, 520)
    assert [annotation.text for annotation in single.layout.annotations] == [
        "first (no_catch)"
    ]
    first_line = next(trace for trace in single.data if trace.name == "first")
    reference_line = next(trace for trace in single.data if trace.name == "reference")
    assert first_line.legendgroup == "gru"
    assert first_line.showlegend is True
    assert reference_line.showlegend is True

    sequence_path = write_velocity_figure(
        (first, second),
        output_dir=tmp_path,
        references=(reference,),
        title="Delayed timing / pre-go hold target-radial velocity ({bank_kind})",
    )
    sequence, sequence_written = writes.pop()
    assert sequence_path == sequence_written == single_path
    assert sequence.layout.title.text == (
        "Delayed timing / pre-go hold target-radial velocity (no_catch)"
    )
    assert (sequence.layout.width, sequence.layout.height) == (980, 520)
    assert [annotation.text for annotation in sequence.layout.annotations] == ["first", "second"]
    assert sequence.layout.yaxis.matches == "y2"
    profile_lines = [trace for trace in sequence.data if trace.name in {"first", "second"}]
    assert [trace.legendgroup for trace in profile_lines] == [
        "run-experiment-first",
        "run-experiment-second",
    ]
    assert all(trace.showlegend is True for trace in profile_lines)
    reference_lines = [trace for trace in sequence.data if trace.name == "reference"]
    assert [trace.showlegend for trace in reference_lines] == [True, False]


def test_velocity_replicate_writer_preserves_single_and_sequence_contracts(
    tmp_path, monkeypatch
) -> None:
    writes: list[tuple[go.Figure, Path]] = []
    monkeypatch.setattr(
        go.Figure,
        "write_html",
        lambda figure, path: writes.append((figure, Path(path))),
    )
    first = _velocity_profile("first", bank_kind="catch")
    second = _velocity_profile("second", bank_kind="catch")
    reference = _velocity_reference()

    single_path = write_velocity_by_replicate_figure(
        first,
        output_dir=tmp_path,
        references=(reference,),
        title="Delayed movement-bank target-radial velocity by replicate ({bank_kind})",
    )
    single, single_written = writes.pop()
    assert single_path == single_written == (
        tmp_path / "forward_velocity_profiles_by_replicate_stochastic.html"
    )
    assert single.layout.title.text == (
        "Delayed movement-bank target-radial velocity by replicate (catch)"
    )
    assert (single.layout.width, single.layout.height) == (940, 560)
    assert [annotation.text for annotation in single.layout.annotations] == [
        "first by replicate (catch)"
    ]
    replicate_lines = [
        trace
        for trace in single.data
        if trace.name.startswith("replicate ") and trace.mode == "lines"
    ]
    assert [trace.legendgroup for trace in replicate_lines] == ["replicate-0", "replicate-1"]
    assert all(trace.showlegend is True for trace in replicate_lines)
    assert single.layout.legend.groupclick == "togglegroup"

    sequence_path = write_velocity_by_replicate_figure(
        (first, second),
        output_dir=tmp_path,
        references=(reference,),
        title=(
            "Delayed timing / pre-go hold target-radial velocity by replicate "
            "({bank_kind})"
        ),
    )
    sequence, sequence_written = writes.pop()
    assert sequence_path == sequence_written == single_path
    assert sequence.layout.title.text == (
        "Delayed timing / pre-go hold target-radial velocity by replicate (catch)"
    )
    assert (sequence.layout.width, sequence.layout.height) == (1020, 560)
    assert [annotation.text for annotation in sequence.layout.annotations] == ["first", "second"]
    assert sequence.layout.yaxis.matches == "y2"
    replicate_lines = [
        trace
        for trace in sequence.data
        if trace.name.startswith("replicate ") and trace.mode == "lines"
    ]
    assert [trace.showlegend for trace in replicate_lines] == [True, True, False, False]
    reference_lines = [trace for trace in sequence.data if trace.name == "reference"]
    assert [trace.showlegend for trace in reference_lines] == [True, False]


def test_velocity_writer_adapter_preserves_input_and_title_contracts(tmp_path) -> None:
    movement = runpy.run_path(
        REPO_ROOT
        / "results/6c36536/scripts/materialize_delayed_movement_bank_velocity_profiles.py"
    )
    calls: list[tuple[object, str]] = []

    def capture(profiles, *, output_dir, references, title):
        del references
        calls.append((profiles, title))
        suffix = "_by_replicate" if "by replicate" in title else ""
        return output_dir / f"forward_velocity_profiles{suffix}_stochastic.html"

    for namespace in (movement,):
        namespace["write_velocity_figure"].__globals__["canonical_write_velocity_figure"] = (
            capture
        )
        namespace["write_velocity_by_replicate_figure"].__globals__[
            "canonical_write_velocity_by_replicate_figure"
        ] = capture
    profile = _velocity_profile("row")

    outputs = (
        movement["write_velocity_figure"](profile, output_dir=tmp_path, references=()),
        movement["write_velocity_by_replicate_figure"](
            profile, output_dir=tmp_path, references=()
        ),
    )

    assert [isinstance(call[0], tuple) for call in calls] == [False, False]
    assert [call[1] for call in calls] == [
        "Delayed movement-bank target-radial velocity ({bank_kind})",
        "Delayed movement-bank target-radial velocity by replicate ({bank_kind})",
    ]
    assert [output.name for output in outputs] == [
        "forward_velocity_profiles_stochastic.html",
        "forward_velocity_profiles_by_replicate_stochastic.html",
    ]


def test_owned_scalar_helpers_are_import_routed_not_redefined() -> None:
    expected_absent = {
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
            "canonical_materialize_nominal_velocity_figure",
        },
        "results/c92ebd8/scripts/materialize_pgd_1p05_nominal_velocity_profiles.py": {
            "canonical_materialize_nominal_velocity_figure"
        },
        "results/c92ebd8/scripts/materialize_pgd_ofb_budget_stabilization_responses.py": {
            "canonical_build_family_figure"
        },
        "results/d55c5f0/scripts/materialize_soft_pgd_stabilization_responses.py": {
            "canonical_build_family_figure"
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


def test_residual_figure_members_are_thin_canonical_adapters() -> None:
    members = {
        "results/c92ebd8/scripts/materialize_pgd_ofb_budget_stabilization_responses.py": (
            "build_family_figure",
            "canonical_build_family_figure",
            60,
        ),
        "results/d55c5f0/scripts/materialize_soft_pgd_stabilization_responses.py": (
            "build_family_figure",
            "canonical_build_family_figure",
            55,
        ),
        "results/c92ebd8/scripts/materialize_pgd_1p05_nominal_velocity_profiles.py": (
            "materialize_figure",
            "canonical_materialize_nominal_velocity_figure",
            55,
        ),
        "results/c92ebd8/scripts/materialize_post_training_figures.py": (
            "materialize_nominal_velocity_profiles",
            "canonical_materialize_nominal_velocity_figure",
            45,
        ),
    }
    forbidden = {
        "build_nominal_profile_figure",
        "build_stabilization_family_figure",
        "profile_comparison_grid",
        "save_figure",
    }
    for relative_path, (name, canonical_name, max_lines) in members.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert canonical_name in calls, (relative_path, canonical_name)
        assert calls.isdisjoint(forbidden), (relative_path, calls & forbidden)
        assert function.end_lineno - function.lineno + 1 <= max_lines, relative_path


def test_delayed_eval_cluster_members_are_thin_canonical_adapters() -> None:
    for relative_path in (
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


def test_velocity_writer_cluster_members_are_thin_canonical_adapters() -> None:
    scripts = (
        "results/6c36536/scripts/materialize_delayed_movement_bank_velocity_profiles.py",
    )
    expected = {
        "write_velocity_figure": "canonical_write_velocity_figure",
        "write_velocity_by_replicate_figure": (
            "canonical_write_velocity_by_replicate_figure"
        ),
    }
    forbidden = {"profile_comparison_grid", "add_band_trace", "add_reference_trace"}
    for relative_path in scripts:
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        definitions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, canonical_name in expected.items():
            function = definitions[name]
            calls = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            assert canonical_name in calls, (relative_path, name)
            assert calls.isdisjoint(forbidden), (relative_path, name, calls & forbidden)
            assert function.end_lineno - function.lineno < 30, (relative_path, name)

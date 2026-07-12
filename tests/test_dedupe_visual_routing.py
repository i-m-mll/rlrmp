from __future__ import annotations

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from rlrmp.viz.figures import (
    build_profile_family_figure,
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
    }
    for relative_path, names in expected_absent.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert definitions.isdisjoint(names), (relative_path, definitions & names)


def test_retired_moderate_and_calibrated_producers_are_absent() -> None:
    for relative_path in (
        "results/3244f1a/scripts/materialize_final_small_bank_profiles.py",
        "results/c92ebd8/scripts/materialize_pgd_1p05_perturbation_response_overlays.py",
        "results/c92ebd8/scripts/materialize_pgd_1p05_moderate_perturbation_profiles_overlay.py",
        "results/c92ebd8/scripts/materialize_post_training_figures.py",
    ):
        assert not (REPO_ROOT / relative_path).exists()


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

from __future__ import annotations

import csv
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

from rlrmp.eval.kinematics import initial_effector_position, initial_effector_velocity
from rlrmp.io import json_ready, write_csv_rows
from rlrmp.viz import add_band_trace, hex_to_rgba


def test_hex_to_rgba_and_band_trace_preserve_plotly_contract() -> None:
    fig = go.Figure()
    x = np.array([0.0, 1.0])
    mean = np.array([1.0, 2.0])
    add_band_trace(
        fig,
        x=x,
        mean=mean,
        spread=np.array([0.25, 0.5]),
        color="#336699",
        name="profile",
    )

    assert hex_to_rgba("#336699", 0.25) == "rgba(51,102,153,0.25)"
    assert len(fig.data) == 2
    assert fig.data[0].fill == "toself"
    assert np.asarray(fig.data[1].y).tolist() == mean.tolist()


def test_csv_and_json_helpers_preserve_declared_columns(tmp_path) -> None:
    path = tmp_path / "nested" / "rows.csv"
    write_csv_rows(path, [{"b": np.int64(2), "a": 1}], fieldnames=("a", "b"))

    with path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{"a": "1", "b": "2"}]
    assert json_ready({"x": np.array([1, 2]), "y": (np.float64(3.0),)}) == {
        "x": [1, 2],
        "y": [3.0],
    }


def test_initial_effector_helpers_support_vector_initial_states() -> None:
    trials = SimpleNamespace(
        inits={"mechanics.vector": jnp.array([[1.0, 2.0, 3.0, 4.0]])}
    )

    np.testing.assert_array_equal(initial_effector_position(trials), [[1.0, 2.0]])
    np.testing.assert_array_equal(initial_effector_velocity(trials), [[3.0, 4.0]])

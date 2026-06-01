"""Tests for the temporary GRU pilot figure materializer."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.gru_pilot_figures import active_loss_term_labels, load_gru_training_history


def _run_spec(*, hidden_weight: float = 0.0) -> dict[str, object]:
    return {
        "hps": {
            "loss": {
                "weights": {
                    "effector_pos_running": 1e6,
                    "effector_terminal_pos": 1e6,
                    "effector_terminal_vel": 1e5,
                    "effector_vel_running": 1e5,
                    "nn_hidden": hidden_weight,
                    "nn_output": 1.0,
                }
            }
        }
    }


def _write_history(path, labels: tuple[str, ...]) -> None:
    with path.open("wb") as stream:
        stream.write(b"null\n")
        for context_offset in (0.0, 100.0):
            for idx, _label in enumerate(labels):
                np.save(stream, np.full((3, 2), context_offset + idx, dtype=np.float64))
                np.save(stream, np.asarray(float(idx + 1), dtype=np.float64))
            np.save(stream, np.asarray(1.0, dtype=np.float64))
        np.save(stream, np.full((3, 2), 0.01, dtype=np.float64))


def test_active_loss_term_labels_follow_gru_feedbax_order() -> None:
    assert active_loss_term_labels(_run_spec(hidden_weight=0.0)) == (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "nn_output",
    )
    assert active_loss_term_labels(_run_spec(hidden_weight=1e-5)) == (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "nn_hidden",
        "nn_output",
    )


def test_load_gru_training_history_rebuilds_feedbax_loss_tree(tmp_path) -> None:
    labels = active_loss_term_labels(_run_spec(hidden_weight=1e-5))
    path = tmp_path / "training_history.eqx"
    _write_history(path, labels)

    history = load_gru_training_history(_run_spec(hidden_weight=1e-5), path)

    assert history.loss.names == labels
    assert history.loss_validation.names == labels
    assert history.loss.children[4].label == "nn_hidden"
    assert np.asarray(history.loss.children[4].value).shape == (3, 2)
    assert float(history.loss.children[4].weight) == 5.0
    assert np.asarray(history.learning_rate).shape == (3, 2)

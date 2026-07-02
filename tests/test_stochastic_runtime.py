"""Tests for Feedbax-facing stochastic runtime wiring."""

from __future__ import annotations

import argparse

import jax.random as jr
import pytest
from feedbax.runtime.noise import CompositeNoise, Multiplicative, Normal
from feedbax.runtime.channel import Channel

from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.model.stochastic_runtime import (
    PLANT_PROCESS_FORCE_NOISE_LABEL,
    stochastic_runtime_config_from_model,
)
from rlrmp.train.minimax import build_hps


def _args(**overrides):
    base = {
        "n_warmup_batches": 1,
        "n_adversary_batches": 1,
        "controller_lr": 0.01,
        "loss_update_enabled": False,
        "loss_update_ratio": 0.3,
        "hidden_type": "gru",
        "sisu_gating": "additive",
        "n_replicates": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_legacy_motor_noise_alias_preserves_feedbax_command_split() -> None:
    hps = build_hps(_args())

    config = stochastic_runtime_config_from_model(hps.model)

    assert config.sensory_noise_std == 0.01
    assert config.signal_dependent_motor_noise_std == 0.01
    assert config.additive_motor_noise_std == pytest.approx(0.018)
    assert config.plant_process_force_noise_std == 0.0


def test_runtime_wires_explicit_command_noise_before_force_filter() -> None:
    hps = build_hps(
        _args(
            sensory_noise_std=0.02,
            additive_motor_noise_std=0.03,
            signal_dependent_motor_noise_std=0.04,
        )
    )

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    efferent = pair.model.nodes["efferent"]
    noise_func = efferent.noise_func

    assert efferent.add_noise is True
    assert isinstance(noise_func, CompositeNoise)
    signal_dependent, additive = noise_func.terms
    assert isinstance(signal_dependent, Multiplicative)
    assert isinstance(signal_dependent.noise_func, Normal)
    assert signal_dependent.noise_func.std == 0.04
    assert isinstance(additive, Normal)
    assert additive.std == 0.03
    assert ("net", "output", "efferent", "input") in _edge_set(pair.model)
    assert ("efferent", "output", "force_filter", "input") in _edge_set(pair.model)


def test_runtime_wires_plant_process_noise_after_intervenor_before_mechanics() -> None:
    hps = build_hps(
        _args(
            additive_motor_noise_std=0.0,
            signal_dependent_motor_noise_std=0.0,
            plant_process_force_noise_std=0.05,
        )
    )

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    process_node = pair.model.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL]

    assert isinstance(process_node, Channel)
    assert process_node.noise_func.std == 0.05
    assert ("force_filter", "output", PLANT_INTERVENOR_LABEL, "force") in _edge_set(pair.model)
    assert (
        PLANT_INTERVENOR_LABEL,
        "force",
        PLANT_PROCESS_FORCE_NOISE_LABEL,
        "input",
    ) in _edge_set(pair.model)
    assert (
        PLANT_PROCESS_FORCE_NOISE_LABEL,
        "output",
        "mechanics",
        "force",
    ) in _edge_set(pair.model)


def test_linear_tracker_runtime_uses_same_stochastic_force_channel() -> None:
    hps = build_hps(
        _args(
            hidden_type="linear_tracker",
            additive_motor_noise_std=0.03,
            signal_dependent_motor_noise_std=0.04,
            plant_process_force_noise_std=0.05,
        )
    )

    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    efferent = pair.model.nodes["efferent"]
    noise_func = efferent.noise_func

    assert pair.model.nodes["net"].__class__.__name__ == "AffineFeedbackController"
    assert pair.model.nodes["reference_mux"].__class__.__name__ == "Mux"
    assert pair.model.nodes["zero_velocity"].__class__.__name__ == "Constant"
    assert "net_state" not in pair.model.nodes
    assert efferent.add_noise is True
    assert isinstance(noise_func, CompositeNoise)
    assert noise_func.terms[0].noise_func.std == 0.04
    assert noise_func.terms[1].std == 0.03
    assert isinstance(pair.model.nodes[PLANT_PROCESS_FORCE_NOISE_LABEL], Channel)
    assert (
        PLANT_PROCESS_FORCE_NOISE_LABEL,
        "output",
        "mechanics",
        "force",
    ) in _edge_set(pair.model)


def _edge_set(model) -> set[tuple[str, str, str, str]]:
    return {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in model.wires
    }

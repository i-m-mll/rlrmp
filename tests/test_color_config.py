from __future__ import annotations

import plotly.colors as plc

import rlrmp
from feedbax.config.namespace import TreeNamespace
from feedbax.plot.color_setup import get_color_config, setup_colors
from feedbax.plugins.registry import ExperimentRegistry
from rlrmp.viz.color_config import (
    RLRMP_COLOR_CONFIG,
    RLRMP_COLOR_CONFIG_SCHEMA_ID,
    RLRMP_COLOR_CONFIG_SCHEMA_VERSION,
    register_rlrmp_color_config,
)


def test_rlrmp_color_config_preserves_project_palette_and_spec_semantics() -> None:
    assert RLRMP_COLOR_CONFIG.schema_id == RLRMP_COLOR_CONFIG_SCHEMA_ID
    assert RLRMP_COLOR_CONFIG.schema_version == RLRMP_COLOR_CONFIG_SCHEMA_VERSION
    assert dict(RLRMP_COLOR_CONFIG.colorscales) == {
        "train__pert__std": "viridis",
        "pert__amp": "plotly3",
        "sisu": "thermal",
        "reach_condition": "phase",
        "replicate": "twilight",
        "trial": "Tealgrn",
        "pert_var": tuple(plc.qualitative.D3),
    }

    hps = TreeNamespace(
        pert=TreeNamespace(amp=(0.0, 0.5, 1.0)),
        train=TreeNamespace(pert=TreeNamespace(std=(0.0, 0.1))),
        sisu=(0.0, 0.25, 1.0),
        eval_n=3,
    )
    assert tuple(RLRMP_COLOR_CONFIG.color_specs["pert__amp"].sequence_fn(hps)) == (
        0.0,
        0.5,
        1.0,
    )
    assert tuple(RLRMP_COLOR_CONFIG.color_specs["train__pert__std"].sequence_fn(hps)) == (
        0.0,
        0.1,
    )
    assert tuple(RLRMP_COLOR_CONFIG.color_specs["sisu"].sequence_fn(hps)) == (0.0, 0.25, 1.0)
    assert tuple(RLRMP_COLOR_CONFIG.color_specs["trial"].sequence_fn(hps)) == (0, 1, 2)

    colors, colorscales = setup_colors(hps, color_config=RLRMP_COLOR_CONFIG)
    assert set(colors) == {"pert__amp", "train__pert__std", "sisu", "trial"}
    assert colorscales["pert__amp"] == "plotly3"
    assert colorscales["train__pert__std"] == "viridis"
    assert colorscales["sisu"] == "thermal"


def test_register_rlrmp_color_config_installs_schema_in_feedbax_registry() -> None:
    registered = register_rlrmp_color_config()

    assert registered is RLRMP_COLOR_CONFIG
    assert get_color_config(RLRMP_COLOR_CONFIG_SCHEMA_ID) is RLRMP_COLOR_CONFIG


def test_register_experiment_package_invokes_live_color_registration(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(rlrmp, "_RECIPES_REGISTERED", None)
    monkeypatch.setattr(
        "rlrmp.viz.color_config.register_rlrmp_color_config",
        lambda: calls.append(RLRMP_COLOR_CONFIG) or RLRMP_COLOR_CONFIG,
    )

    rlrmp.register_experiment_package(ExperimentRegistry())

    assert calls == [RLRMP_COLOR_CONFIG]

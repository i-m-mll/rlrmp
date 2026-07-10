"""Model-derived GraphSpec export parity across active builder variants (7811e47).

For every active builder variant, materialize a runtime model, export its
GraphSpec from the constructed model via ``graph_spec_from_model`` (point-mass
variants go through ``setup_task_model_pair`` / a materialized point-mass graph;
CS-LSS variants through ``materialize_cs_lss_gru_graph_spec``), and assert the
exported spec (a) round-trips through the Feedbax GraphSpec JSON contract with
byte-for-byte payload parity and (b) re-materializes to a runtime ``Graph``.

A deliberate-mismatch negative case proves the parity check has teeth, and a
node-type parity check confirms the model-derived export reproduces the
requested builder's component contract.
"""

from __future__ import annotations

import argparse
import json

import jax.random as jr
import pytest

# Fully initialize the rlrmp.analysis/train recipe chain before importing
# rlrmp.model.cs_lss_gru names directly, to avoid a partial-initialization
# circular import (cs_lss_gru -> analysis -> ... -> task_model -> cs_lss_gru).
import rlrmp.analysis.math.cs_game_card as _cs_game_card  # noqa: F401  (import-for-side-effect)
from feedbax.contracts.graph import GraphSpec
from feedbax.runtime.graph import Graph

from rlrmp.model.cs_lss_gru import (
    build_cs_lss_gru_graph_spec,
    materialize_cs_lss_gru_graph_spec,
)
from rlrmp.model.feedbax_graph import (
    build_point_mass_sensorimotor_graph_spec,
    build_rlrmp_feedbax_graph_bundle,
    graph_spec_from_model,
    graph_spec_payload,
    materialize_rlrmp_graph_spec,
)
from rlrmp.train.minimax_native import build_hps
from rlrmp.train.task_model import build_task_base, setup_task_model_pair


pytestmark = pytest.mark.feedbax_contract


def _args(**overrides):
    base = {
        "n_warmup_batches": 10,
        "n_adversary_batches": 20,
        "controller_lr": 0.01,
        "loss_update_enabled": False,
        "loss_update_ratio": 0.3,
        "hidden_type": "gru",
        "sisu_gating": "additive",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _hps(**overrides):
    hps = build_hps(_args(**overrides))
    if hps.pert.type == "gusts":
        hps = hps | {"pert": hps.pert | {"type": "constant"}}
    return hps


def _unmasked_population() -> argparse.Namespace:
    return argparse.Namespace(
        n_input_only=0,
        n_readout_only=0,
        n_recurrent_only=0,
        n_input_readout=0,
    )


# --------------------------------------------------------------------------- #
# Per-variant model-derived export builders
# --------------------------------------------------------------------------- #


def _point_mass_export(*, n_replicates: int = 2, **hps_overrides):
    hps = _hps(n_replicates=n_replicates, **hps_overrides)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    exported = graph_spec_from_model(pair.model, n_replicates=int(hps.model.n_replicates))
    return exported, materialize_rlrmp_graph_spec


def _population_gru_export():
    hps = _hps(hidden_type="gru")
    task = build_task_base(hps)
    spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        task=task,
        n_extra_inputs=1,
        population_structure=_unmasked_population(),
        hidden_type=hps.hidden_type,
        sisu_gating="additive",
    )
    model = materialize_rlrmp_graph_spec(spec)
    exported = graph_spec_from_model(model)
    return exported, materialize_rlrmp_graph_spec


def _cs_lss_export(*, initial_hidden_encoder: bool):
    kwargs = dict(
        hidden_size=5,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        key=jr.PRNGKey(7),
    )
    if initial_hidden_encoder:
        kwargs["initial_hidden_encoder"] = True
    else:
        kwargs["force_filter_feedback"] = True
    spec = build_cs_lss_gru_graph_spec(**kwargs)
    model = materialize_cs_lss_gru_graph_spec(spec)
    exported = graph_spec_from_model(model)
    return exported, materialize_cs_lss_gru_graph_spec


_VARIANT_BUILDERS = {
    "point_mass_gru": lambda: _point_mass_export(hidden_type="gru"),
    # Single-replicate export regression (74af1ef): n_replicates=1 ensembles
    # kept a spurious leading singleton axis in the efferent channel input_proto
    # ([1, 2] vs the [2] prototype) and failed re-materialization.
    "point_mass_gru_n1": lambda: _point_mass_export(hidden_type="gru", n_replicates=1),
    "linear": lambda: _point_mass_export(hidden_type="linear"),
    "linear_tracker": lambda: _point_mass_export(hidden_type="linear_tracker"),
    "minimax_adversary": lambda: _point_mass_export(
        hidden_type="gru", adversary_type="linear_dynamics"
    ),
    # Multiplicative-SISU export (0f67665): the multiplicative sisu_modulator ->
    # cell.hidden wire carries a `constant` recurrent initializer whose value is
    # now emitted JSON-native ([0.0]*hidden_size + dtype name), so the stacked
    # export round-trips and re-materializes like every other row.
    "multiplicative_sisu": lambda: _point_mass_export(
        hidden_type="gru", sisu_gating="multiplicative"
    ),
    "population_gru": _population_gru_export,
    "cs_lss": lambda: _cs_lss_export(initial_hidden_encoder=False),
    "cs_lss_initial_hidden": lambda: _cs_lss_export(initial_hidden_encoder=True),
}


def _assert_export_round_trip(exported: GraphSpec, rematerialize) -> None:
    payload = graph_spec_payload(exported)
    round_tripped = GraphSpec.model_validate_json(json.dumps(payload))
    assert graph_spec_payload(round_tripped) == payload
    assert isinstance(rematerialize(round_tripped), Graph)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", sorted(_VARIANT_BUILDERS))
def test_model_derived_export_round_trips_and_rematerializes(variant: str) -> None:
    exported, rematerialize = _VARIANT_BUILDERS[variant]()
    _assert_export_round_trip(exported, rematerialize)


def test_point_mass_gru_export_matches_requested_component_contract() -> None:
    hps = _hps(hidden_type="gru", n_replicates=2)
    requested = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
    ).graph_spec
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    exported = graph_spec_from_model(pair.model, n_replicates=int(hps.model.n_replicates))

    requested_types = {name: node.type for name, node in requested.nodes.items()}
    exported_types = {name: node.type for name, node in exported.nodes.items()}
    assert exported_types == requested_types


def test_export_parity_negative_rejects_corrupted_component_type() -> None:
    exported, rematerialize = _point_mass_export(hidden_type="gru")
    bad_nodes = dict(exported.nodes)
    bad_nodes["mechanics"] = exported.nodes["mechanics"].model_copy(
        update={"type": "TotallyBogusComponentType"}
    )
    corrupted = exported.model_copy(update={"nodes": bad_nodes})

    payload = graph_spec_payload(corrupted)
    round_tripped = GraphSpec.model_validate_json(json.dumps(payload))
    with pytest.raises(Exception):
        rematerialize(round_tripped)

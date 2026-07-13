"""Focused contracts for canonical linear-recurrent C&S training bases."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
from rlrmp.train.linear_recurrent_native import (
    LINEAR_RECURRENT_ARCHITECTURE,
    LINEAR_RECURRENT_CERTIFICATE_MODE,
    LINEAR_RECURRENT_KERNEL_OWNER,
    LINEAR_RECURRENT_NATIVE_METHOD,
    LINEAR_RECURRENT_RUNNER,
    author_linear_recurrent_training_base_from_canonical,
    linear_recurrent_architecture_metadata,
)
from rlrmp.train.execution_preparation import _runtime_config
from rlrmp.train.training_base_routes import route_training_base


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BASE_PATH = REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"


def _canonical_base() -> TrainingRunSpec:
    register_rlrmp_training_methods()
    payload = json.loads(CANONICAL_BASE_PATH.read_text(encoding="utf-8"))
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def test_linear_recurrent_base_preserves_canonical_cs_task_and_registered_method() -> None:
    base = _canonical_base()
    authored = author_linear_recurrent_training_base_from_canonical(
        base, training_distribution="nominal"
    )
    graph = GraphSpec.model_validate(authored.graph.inline)
    cell = graph.subgraphs["net"].nodes["cell"]
    recurrent_graph = graph.subgraphs["net"]

    assert authored.task == base.task
    assert authored.objective == base.objective
    assert (
        graph.nodes["mechanics"] == GraphSpec.model_validate(base.graph.inline).nodes["mechanics"]
    )
    assert authored.method_ref.key == "rlrmp/cs_supervised/v1"
    assert authored.training_config.network_type == LINEAR_RECURRENT_ARCHITECTURE
    assert cell.type == "VanillaRNN"
    assert cell.params["activation"] == "identity"
    assert cell.params["use_bias"] is False
    assert recurrent_graph.nodes["readout"].params["use_bias"] is False
    assert "h0_encoder" not in recurrent_graph.nodes
    assert "h0_context" not in recurrent_graph.input_ports
    assert "h0_context" not in recurrent_graph.input_bindings
    hidden_initializer = next(
        wire.recurrent_initializer
        for wire in recurrent_graph.wires
        if wire.source_node == "cell"
        and wire.source_port == "hidden"
        and wire.target_node == "cell"
        and wire.target_port == "hidden"
        and wire.temporality == "recurrent"
    )
    assert hidden_initializer["kind"] == "zeros"
    assert hidden_initializer["shape"] == [cell.params["hidden_size"]]
    assert authored.metadata["architecture"] == LINEAR_RECURRENT_ARCHITECTURE
    assert authored.metadata["native_method"] == "rlrmp/cs_supervised/v1"
    assert authored.method_extensions.metadata["runner"] == "rlrmp.train.orchestrated_row"
    for metadata in (
        authored.method_payload.metadata,
        authored.graph.metadata,
        authored.method_extensions.metadata,
        authored.worker_execution.metadata,
    ):
        assert metadata["architecture"] == LINEAR_RECURRENT_ARCHITECTURE
        assert metadata["controller_architecture"] == LINEAR_RECURRENT_ARCHITECTURE


def test_linear_recurrent_nominal_and_robust_bases_share_identity_contracts() -> None:
    base = _canonical_base()
    nominal = author_linear_recurrent_training_base_from_canonical(
        base, training_distribution="nominal"
    )
    robust = author_linear_recurrent_training_base_from_canonical(
        base, training_distribution="broad_epsilon_pgd"
    )

    assert nominal.method_payload.payload.get("pre_step") is None
    assert robust.method_payload.payload["pre_step"]["kind"] == "broad_epsilon_pgd"
    assert nominal.checkpoint_progress.checkpoint_interval == (
        base.checkpoint_progress.checkpoint_interval
    )
    assert nominal.checkpoint_progress.progress_interval == (
        base.checkpoint_progress.progress_interval
    )
    assert robust.artifacts == base.artifacts
    assert nominal.method_ref == robust.method_ref
    assert nominal.checkpoint_progress.continuation is None
    assert nominal.checkpoint_progress.resume_from is None
    assert nominal.worker_execution.resume is None
    assert nominal.method_payload.payload["config"]["resume"] is False
    assert nominal.method_payload.payload["config"]["allow_fresh_start"] is True
    assert training_spec_sha256(nominal.model_dump(mode="json", exclude_none=True)) != (
        training_spec_sha256(robust.model_dump(mode="json", exclude_none=True))
    )


def test_adaptive_epsilon_source_metadata_is_normalized_to_cs_supervised() -> None:
    base = _canonical_base()
    assert base.metadata["native_method"] == "rlrmp/adaptive_epsilon_curriculum/v1"
    assert base.worker_execution.metadata["kernel_owner"] == ("rlrmp.train.adaptive_epsilon_native")
    assert base.method_extensions.metadata["runner"] == "rlrmp.train.cs_nominal_gru"

    authored = author_linear_recurrent_training_base_from_canonical(base)

    assert authored.method_ref.key == LINEAR_RECURRENT_NATIVE_METHOD
    assert authored.metadata["native_method"] == LINEAR_RECURRENT_NATIVE_METHOD
    assert authored.metadata["runner"] == LINEAR_RECURRENT_RUNNER
    assert authored.metadata["architecture"] == LINEAR_RECURRENT_ARCHITECTURE
    for metadata in (
        authored.method_payload.metadata,
        authored.graph.metadata,
        authored.method_extensions.metadata,
        authored.worker_execution.metadata,
    ):
        assert metadata["native_method"] == LINEAR_RECURRENT_NATIVE_METHOD
        assert metadata["runner"] == LINEAR_RECURRENT_RUNNER
        assert metadata["architecture"] == LINEAR_RECURRENT_ARCHITECTURE
        assert metadata["controller_architecture"] == LINEAR_RECURRENT_ARCHITECTURE
    assert authored.worker_execution.metadata["kernel_owner"] == (LINEAR_RECURRENT_KERNEL_OWNER)
    assert authored.worker_execution.metadata["native_executor"] == (
        "feedbax.training.executor.execute_training_run_spec"
    )


def test_linear_recurrent_handoff_uses_exact_standard_certificate_vocabulary() -> None:
    authored = author_linear_recurrent_training_base_from_canonical(_canonical_base())
    metadata = linear_recurrent_architecture_metadata(authored)

    assert metadata["architecture"] == "linear_recurrence"
    assert metadata["certificate_mode"] == "augmented_linear"
    assert metadata["component_provider"] == "rlrmp.eval.linear_recurrent_augmented"
    assert "component_kwargs" not in metadata
    assert set(metadata["component_input_contract"]) == {
        "augmented_states",
        "candidate_augmented_action_sensitivity",
        "reference_augmented_action_sensitivity",
        "candidate_transition",
        "reference_transition",
        "candidate_value_matrices",
        "reference_value_matrices",
        "bellman_hessian",
        "recurrence_diagnostics",
    }
    contract = authored.metadata["certificate_contract"]
    assert contract["static_gain_coercion"] == "forbidden"
    assert contract["augmented_state_basis"] == [
        "controller_visible_target_relative_post_step_coupled_state",
        "previous_step_hidden_state",
    ]
    assert contract["state_history_timing"] == "feedbax_post_step_history_pair"
    assert contract["recurrence"] == "zero_bias_leaky_identity_activation"
    construction = contract["component_inputs"]["candidate_transition"]["construction"]
    assert "A_target_relative + B @ K_x" in construction
    assert "alpha * cell.weight_ih" in construction


def test_linear_recurrent_runtime_uses_zero_bias_identity_cell() -> None:
    authored = author_linear_recurrent_training_base_from_canonical(_canonical_base())
    _args, hps = _runtime_config(dict(authored.method_payload.payload["config"]))

    assert isinstance(hps.hidden_type, partial)
    assert hps.hidden_type.func.__name__ == "LeakyRNNCell"
    assert hps.model.initial_hidden_encoder is False
    cell = hps.hidden_type(input_size=3, hidden_size=4, key=jr.PRNGKey(0))
    assert cell.use_bias is False
    assert cell.bias is None
    probe = jnp.asarray([-1.0, 0.0, 2.0])
    state = jnp.asarray([0.5, -0.25, 1.0, -1.0])
    expected = cell.weight_ih @ probe + cell.weight_hh @ state
    assert jnp.allclose(cell(probe, state), expected)


def test_linear_recurrent_route_uses_row_local_manifest_custody() -> None:
    authored = author_linear_recurrent_training_base_from_canonical(_canonical_base())
    routed = route_training_base(
        authored,
        issue="427d0d8",
        row_id="linear-recurrence-test",
    )

    assert routed.artifacts.artifact_root == ("_artifacts/427d0d8/runs/linear-recurrence-test")
    assert routed.artifacts.manifest_root == (
        "_artifacts/427d0d8/runs/linear-recurrence-test/manifests"
    )


def test_linear_recurrent_graph_and_payload_round_trip_without_expanded_patching() -> None:
    authored = author_linear_recurrent_training_base_from_canonical(
        _canonical_base(), training_distribution="broad_epsilon_pgd"
    )
    round_trip = TrainingRunSpec.model_validate(authored.model_dump(mode="json", exclude_none=True))

    assert round_trip.metadata["controller_architecture"] == "linear_recurrence"
    assert round_trip.metadata["certificate_mode"] == LINEAR_RECURRENT_CERTIFICATE_MODE
    assert round_trip.method_payload.payload["config"]["controller_architecture"] == (
        "linear_recurrence"
    )
    assert round_trip.metadata["serialize_do_not_rederive"] is True

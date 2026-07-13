"""Focused contract tests for canonical static-linear training bases."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax.random as jr
import pytest
from feedbax.contracts.graph import GraphSpec
from feedbax.training.manifest_preflight import build_training_run_manifest_spec_payloads
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.model.cs_lss_static_linear import (
    STATIC_LINEAR_CONTROLLER_KIND,
    build_cs_lss_static_linear_graph_spec,
)
from rlrmp.runtime.training_run_specs import register_rlrmp_cs_supervised_method
from rlrmp.train.adaptive_epsilon_native import (
    ensure_adaptive_epsilon_training_method_registered,
)
from rlrmp.train.static_linear_native import (
    STATIC_LINEAR_BRIDGE_ARCHITECTURE,
    STATIC_LINEAR_CERTIFICATE_MODE,
    STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS,
    STATIC_LINEAR_FEEDBACK_BASIS,
    STATIC_LINEAR_NATIVE_METHOD,
    STATIC_LINEAR_RUNNER,
    author_static_linear_training_base,
    author_static_linear_training_base_from_canonical,
    static_linear_architecture_metadata,
    static_linear_runtime_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _register_source_and_target_methods() -> None:
    ensure_adaptive_epsilon_training_method_registered()
    register_rlrmp_cs_supervised_method()


def test_static_linear_graph_is_gain_only_and_keeps_epsilon_input() -> None:
    graph = build_cs_lss_static_linear_graph_spec(
        input_size=1,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        force_filter_feedback=True,
        key=jr.PRNGKey(0),
    )

    assert graph.nodes["net"].type == "AffineFeedbackController"
    assert set(graph.nodes["net"].params) == {"gain", "schedule_policy"}
    assert graph.nodes["net"].output_ports == ["command"]
    assert graph.input_bindings["epsilon"] == ("mechanics", "epsilon")
    assert graph.input_bindings["input"] == ("task_input_sink", "input")
    assert graph.subgraphs == {}
    assert [channel["transform"] for channel in graph.nodes["feedback"].params["channels"]] == [
        "target_minus",
        "negate",
        "identity",
    ]


def test_static_linear_preparation_lowers_architecture_without_changing_base_config() -> None:
    config = {
        "controller_architecture": STATIC_LINEAR_CONTROLLER_KIND,
        "n_train_batches": 2,
        "batch_size": 2,
        "n_replicates": 1,
        "hidden_size": 4,
        "output_dir": "_artifacts/427d0d8/runs/static-linear-test",
        "issue": "427d0d8",
    }
    _args, hps = static_linear_runtime_config(config)
    assert hps.hidden_type == STATIC_LINEAR_CONTROLLER_KIND


def test_static_linear_base_preserves_registered_method_and_certificate_handoff() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    graph = build_cs_lss_static_linear_graph_spec(
        input_size=1,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        force_filter_feedback=True,
        key=jr.PRNGKey(0),
    )

    authored = author_static_linear_training_base(
        base,
        graph_spec=graph,
        training_distribution="broad_epsilon_pgd",
    )

    assert authored.method_ref.key == "rlrmp/cs_supervised/v1"
    assert authored.training_config.network_type == STATIC_LINEAR_CONTROLLER_KIND
    assert authored.objective == base.objective
    assert authored.method_payload.payload["pre_step"]["kind"] == "broad_epsilon_pgd"
    assert authored.metadata["controller_architecture"] == STATIC_LINEAR_BRIDGE_ARCHITECTURE
    assert authored.metadata["controller_kind"] == STATIC_LINEAR_CONTROLLER_KIND
    assert authored.metadata["certificate_mode"] == STATIC_LINEAR_CERTIFICATE_MODE
    assert authored.metadata["training_distribution"] == "broad_epsilon"
    assert authored.metadata["training_method_distribution"] == "broad_epsilon_pgd"
    assert authored.metadata["native_method"] == authored.method_ref.key
    assert authored.metadata["native_method"] == STATIC_LINEAR_NATIVE_METHOD
    assert authored.metadata["runner"] == STATIC_LINEAR_RUNNER
    for metadata in (
        authored.method_payload.metadata,
        authored.graph.metadata,
        authored.method_extensions.metadata,
        authored.worker_execution.metadata,
    ):
        assert metadata["controller_architecture"] == STATIC_LINEAR_BRIDGE_ARCHITECTURE
        assert metadata["controller_kind"] == STATIC_LINEAR_CONTROLLER_KIND
        assert metadata["native_method"] == STATIC_LINEAR_NATIVE_METHOD
        assert metadata["runner"] == STATIC_LINEAR_RUNNER
    assert authored.worker_execution.metadata["kernel_owner"] == ("rlrmp.train.cs_nominal_gru")
    assert authored.metadata["certificate_contract"] == {
        "architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
        "mode": STATIC_LINEAR_CERTIFICATE_MODE,
        "training_distribution": "broad_epsilon",
        "state_basis": "coupled_closed_loop_state",
        "action_basis": STATIC_LINEAR_FEEDBACK_BASIS,
        "trainable_paths": ["nodes.net.gain"],
        "memory": "none",
        "required_component_inputs": list(STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS),
        "candidate_gain_source": {
            "parent_role": "training_checkpoint_custody",
            "slot": "model",
            "trainable_path": "nodes.net.gain",
        },
    }


def test_static_linear_nominal_base_has_no_adversarial_pre_step() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    graph = build_cs_lss_static_linear_graph_spec(
        input_size=1,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        force_filter_feedback=True,
        key=jr.PRNGKey(0),
    )
    authored = author_static_linear_training_base(
        base,
        graph_spec=graph,
        training_distribution="nominal",
    )

    assert authored.method_payload.payload.get("pre_step") is None
    assert authored.metadata["training_distribution"] == "nominal"


def test_static_linear_base_is_a_fresh_run_not_an_inherited_continuation() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    assert base.checkpoint_progress.continuation is not None
    assert "source_checkpoint_root" in base.metadata

    authored = author_static_linear_training_base_from_canonical(base)

    assert authored.checkpoint_progress.resume_from is None
    assert authored.checkpoint_progress.checkpoint_slots is None
    assert authored.checkpoint_progress.continuation is None
    assert authored.checkpoint_progress.metadata == {}
    assert authored.worker_execution.resume is None
    assert authored.worker_execution.checkpoint_slots is None
    assert authored.worker_execution.progress is None
    assert authored.method_payload.payload["config"]["resume"] is False
    assert authored.method_payload.payload["config"]["allow_fresh_start"] is True
    assert "source_checkpoint_root" not in authored.metadata
    assert "source_checkpoint_transaction_id" not in authored.metadata
    assert "lr_continuation_schedule" not in authored.metadata


def test_static_linear_base_rejects_noncanonical_feedback_basis() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    graph = build_cs_lss_static_linear_graph_spec(
        target_relative_feedback=False,
        force_filter_feedback=False,
        key=jr.PRNGKey(0),
    )

    with pytest.raises(ValueError, match="canonical static-linear gain"):
        author_static_linear_training_base(
            base,
            graph_spec=graph,
            training_distribution="nominal",
        )


def test_static_linear_base_can_be_emitted_from_canonical_authored_base() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )

    authored = author_static_linear_training_base_from_canonical(base)

    graph = GraphSpec.model_validate(authored.graph.inline)
    assert graph.nodes["net"].type == "AffineFeedbackController"
    assert authored.metadata["controller_architecture"] == STATIC_LINEAR_BRIDGE_ARCHITECTURE


def test_static_linear_metadata_survives_standard_training_manifest_preflight() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    authored = author_static_linear_training_base_from_canonical(
        base,
        training_distribution="broad_epsilon_pgd",
    )

    payloads = build_training_run_manifest_spec_payloads(authored)
    assert payloads.training_spec.inline is not None
    manifest_metadata = payloads.training_spec.inline["metadata"]
    assert manifest_metadata["controller_architecture"] == "time_constrained_free_gain"
    assert manifest_metadata["certificate_mode"] == "static_gain"
    assert manifest_metadata["training_distribution"] == "broad_epsilon"
    assert manifest_metadata["certificate_contract"]["required_component_inputs"] == list(
        STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS
    )
    assert manifest_metadata["certificate_contract"]["candidate_gain_source"] == {
        "parent_role": "training_checkpoint_custody",
        "slot": "model",
        "trainable_path": "nodes.net.gain",
    }


def test_static_linear_handoff_uses_bridge_vocabulary_without_payload_patching() -> None:
    _register_source_and_target_methods()
    base = TrainingRunSpec.model_validate(
        json.loads(
            (REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json").read_text(
                encoding="utf-8"
            )
        )["feedbax_training_run_spec"]
    )
    authored = author_static_linear_training_base_from_canonical(base)

    assert static_linear_architecture_metadata(authored) == {
        "architecture": "time_constrained_free_gain",
        "controller_architecture": "time_constrained_free_gain",
        "controller_kind": "static_linear",
        "certificate_mode": "static_gain",
        "training_distribution": "nominal",
        "certificate_contract": authored.metadata["certificate_contract"],
    }


def test_static_linear_emitter_registers_source_and_target_methods(tmp_path: Path) -> None:
    output = tmp_path / "static-linear.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/emit_static_linear_training_base.py"),
            "--base",
            str(REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"),
            "--output",
            str(output),
            "--training-distribution",
            "broad_epsilon_pgd",
            "--issue",
            "427d0d8",
            "--row-id",
            "static-test",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    emitted = json.loads(output.read_text(encoding="utf-8"))
    assert emitted["metadata"]["controller_architecture"] == "time_constrained_free_gain"
    assert emitted["metadata"]["training_distribution"] == "broad_epsilon"
    assert emitted["metadata"]["native_method"] == "rlrmp/cs_supervised/v1"
    assert emitted["metadata"]["runner"] == "rlrmp.train.orchestrated_row"
    assert emitted["method_extensions"]["metadata"]["native_method"] == ("rlrmp/cs_supervised/v1")
    assert "continuation" not in emitted["checkpoint_progress"]
    assert "resume_from" not in emitted["checkpoint_progress"]
    assert "source_checkpoint_root" not in emitted["metadata"]
    assert "source_checkpoint_transaction_id" not in emitted["metadata"]
    assert emitted["artifacts"]["artifact_root"] == ("_artifacts/427d0d8/runs/static-test")
    assert emitted["artifacts"]["manifest_root"] == (
        "_artifacts/427d0d8/runs/static-test/manifests"
    )

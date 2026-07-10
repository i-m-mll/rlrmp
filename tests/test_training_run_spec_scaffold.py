"""Shared TrainingRunSpec scaffold regression tests."""

from __future__ import annotations

from pathlib import Path

from feedbax.contracts.training import (
    GraphTopologySourceSpec,
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)

from rlrmp.runtime.training_run_specs import build_training_run_spec_scaffold


def test_training_run_spec_scaffold_builds_shared_policies_and_run_spec(
    tmp_path: Path,
) -> None:
    scaffold = build_training_run_spec_scaffold(
        risk_metadata={"source": "test"},
        execution_mode="dry_run",
        require_review=True,
        allow_cloud=False,
        execution_metadata={"entrypoint": "test"},
        artifact_root=str(tmp_path / "artifacts"),
        artifact_metadata={"tracked_spec": "results/test/run.json"},
        checkpoint_interval=5,
        progress_interval=2,
        checkpoint_metadata={"latest_pointer": "latest.json"},
        metadata={"composed_with": "test_payload"},
    )
    contract = standard_supervised_method_contract()
    spec = scaffold.build(
        graph=GraphTopologySourceSpec(
            inline={"nodes": {}, "wires": [], "input_ports": [], "output_ports": []}
        ),
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        method_extensions={},
        method_contract=contract,
        effective_phase=standard_supervised_effective_phase_spec(),
        worker_metadata={"native_executor": "test.executor"},
        metadata={"serialize_do_not_rederive": False, "row": "smoke"},
    )

    assert spec.risk_aggregation.metadata == {"source": "test"}
    assert spec.execution.mode == "dry_run"
    assert spec.artifacts.manifest_root == "_artifacts/feedbax_runs"
    assert spec.checkpoint_progress.checkpoint_interval == 5
    assert spec.worker_execution.metadata == {"native_executor": "test.executor"}
    assert spec.metadata == {
        "row": "smoke",
        "composed_with": "test_payload",
        "serialize_do_not_rederive": True,
    }


def test_scaffold_constructor_is_the_only_owned_top_level_policy_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_source = (repo_root / "src/rlrmp/runtime/training_run_specs.py").read_text(
        encoding="utf-8"
    )
    minimax_source = (repo_root / "src/rlrmp/train/minimax_native/method.py").read_text(
        encoding="utf-8"
    )

    assert runtime_source.count("TrainingRunSpec(") == 1
    assert minimax_source.count("TrainingRunSpec(") == 0
    for constructor in (
        "RiskAggregationSpec(",
        "ExecutionPolicySpec(",
        "ArtifactPolicySpec(",
        "CheckpointProgressPolicySpec(",
    ):
        assert runtime_source.count(constructor) == 1
        assert constructor not in minimax_source

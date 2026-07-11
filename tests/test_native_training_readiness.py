from __future__ import annotations

import inspect
import io
import json
import sys
from collections.abc import Callable

import jax.numpy as jnp
import pytest
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)

from rlrmp.train.executor import cs_supervised


class _BufferedStdoutProbe(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


@pytest.mark.parametrize(
    ("entrypoint", "phase"),
    [
        (cs_supervised._run_cs_supervised_native_from_context, "train_chunk"),
        (
            cs_supervised._run_adaptive_epsilon_native_from_context,
            "adaptive_epsilon_train_chunk",
        ),
        (
            cs_supervised._run_policy_adversary_native_from_context,
            "policy_adversary_train_chunk",
        ),
    ],
)
def test_native_entrypoints_use_shared_readiness_observers(
    entrypoint: Callable[..., object],
    phase: str,
) -> None:
    source = inspect.getsource(entrypoint)

    assert "_execute_native_training_run_spec(" in source
    assert f'progress_phase="{phase}"' in source
    assert "total_batches=int(args.n_train_batches)" in source


@pytest.mark.parametrize(
    "phase",
    ["train_chunk", "adaptive_epsilon_train_chunk", "policy_adversary_train_chunk"],
)
def test_native_observer_configuration_emits_batch_progress(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
) -> None:
    events_dir = tmp_path / "events"
    monkeypatch.setenv("FEEDBAX_RUN_SET_ID", "stage-2")
    monkeypatch.setenv("FEEDBAX_ROW_ID", phase)
    monkeypatch.setenv("FEEDBAX_RUN_EVENTS_DIR", str(events_dir))

    def fake_execute(training_spec, *, progress_callback, run_event_emitter, **kwargs):
        del training_spec, kwargs
        assert run_event_emitter is not None
        progress_callback(
            {
                "coordinate": {"phase": phase, "program_step": 1},
                "metrics": {"train_loss": 1.0},
            }
        )
        return object()

    monkeypatch.setattr(cs_supervised, "execute_training_run_spec", fake_execute)

    cs_supervised._execute_native_training_run_spec(
        object(),
        progress_phase=phase,
        total_batches=2,
    )

    assert capsys.readouterr().out.startswith(f"BATCH phase={phase} batch=0/2")


def test_real_executor_emits_ready_and_flushes_batch_progress_with_buffered_stdout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_dir = tmp_path / "events"
    monkeypatch.setenv("FEEDBAX_RUN_SET_ID", "stage-2")
    monkeypatch.setenv("FEEDBAX_ROW_ID", "flat")
    monkeypatch.setenv("FEEDBAX_RUN_EVENTS_DIR", str(events_dir))
    buffered_stdout = _BufferedStdoutProbe()
    monkeypatch.setattr(sys, "stdout", buffered_stdout)

    result = cs_supervised._execute_native_training_run_spec(
        _minimal_training_spec(),
        progress_phase="train_batch",
        total_batches=1,
        run_id="buffered-readiness",
        initial_slots={
            "model": jnp.array([0.0]),
            "optimizer": {"count": jnp.array([1.0])},
            "prng": jnp.array([0, 1], dtype=jnp.uint32),
            "batch_counter": jnp.array(0, dtype=jnp.int32),
        },
        manifest_root=tmp_path / "manifests",
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert result.final_coordinate.program_step == 1
    assert buffered_stdout.getvalue().startswith("BATCH phase=train_batch batch=0/1")
    assert buffered_stdout.flush_count >= 1
    event_paths = list(events_dir.glob("*.events.jsonl"))
    assert len(event_paths) == 1
    events = [json.loads(line) for line in event_paths[0].read_text().splitlines()]
    assert events[0]["type"] == "ready"
    assert events[0]["payload"]["phase"] == "train_batch"
    assert any(event["type"] == "progress" for event in events)


def _minimal_training_spec() -> TrainingRunSpec:
    return TrainingRunSpec(
        graph={
            "inline": {
                "nodes": {
                    "gain": {
                        "type": "Gain",
                        "params": {"gain": 1.0},
                        "input_ports": ["input"],
                        "output_ports": ["output"],
                    }
                },
                "wires": [],
                "input_ports": ["input"],
                "output_ports": ["output"],
                "input_bindings": {"input": ("gain", "input")},
                "output_bindings": {"output": ("gain", "output")},
            }
        },
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(
                type="target_state",
                label="target",
                selector="port:gain.output",
                target_value=[0.0],
            )
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )

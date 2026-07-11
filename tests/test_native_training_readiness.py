from __future__ import annotations

import inspect
import json
from collections.abc import Callable

import pytest

from rlrmp.train.executor import cs_supervised


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
def test_native_readiness_observers_emit_ready_and_batch_progress(
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
        run_event_emitter.emit("ready", {"phase": phase})
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

    event_paths = list(events_dir.glob("*.events.jsonl"))
    assert len(event_paths) == 1
    events = [json.loads(line) for line in event_paths[0].read_text().splitlines()]
    assert [event["type"] for event in events] == ["ready"]
    assert events[0]["payload"]["phase"] == phase
    assert capsys.readouterr().out.startswith(f"BATCH phase={phase} batch=0/2")

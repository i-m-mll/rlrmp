"""Gate-3 real-row construction checks without training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
)
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training.checkpoint_custody import write_checkpoint_transaction

from rlrmp.train.launch import (
    LaunchRow,
    _PreparedExecution,
    verify_resume_authored_training_intent,
)
from rlrmp.train.fixture_orchestration import (
    fixture_effective_phase_spec,
    fixture_method_contract,
    fixture_method_payload,
    fixture_method_ref,
    register_fixture_method,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_adaptive_continuation_rows_construct_execution_preparations() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # JAX's x64 setting is process-global. The full suite contains analysis tests
    # that deliberately enable it, so execute this training precondition in the
    # same fresh-process boundary used by the repo's other x64-sensitive tests.
    env["JAX_ENABLE_X64"] = "False"
    code = f"""
import json
from pathlib import Path

from rlrmp.train.launch import load_authored_training_intent, prepare_authored_training_rows

repo_root = Path({str(REPO_ROOT)!r})
launch = load_authored_training_intent(
    repo_root / "results/c6c5997/runs/matrix.json", repo_root=repo_root
)
evidence = prepare_authored_training_rows(launch)
print("GATE3_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("GATE3_EVIDENCE=")
    )
    evidence = json.loads(evidence_line.removeprefix("GATE3_EVIDENCE="))
    assert len(evidence) == 3
    for row in evidence:
        assert {"model", "optimizer", "prng", "completed_batches"} <= set(row["slot_names"])
        assert row["has_kernel_context"]
        assert row["has_loss_service"]


def test_strict_resume_loads_a_real_toy_custody_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_fixture_method()
    contract = fixture_method_contract()
    spec = TrainingRunSpec(
        graph={"inline": {"nodes": {}, "wires": [], "input_ports": [], "output_ports": []}},
        task=TaskSpec(type="ToyTask", params={"n_steps": 1}),
        training_config=TrainingConfig(n_batches=1, batch_size=1),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=fixture_method_ref(),
        method_payload=fixture_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=fixture_effective_phase_spec(),
        ),
        checkpoint_progress={
            "checkpoint_interval": 1,
            "metadata": {"checkpoint_dir": str(tmp_path / "target")},
        },
    )
    slots = {"model": 0, "optimizer": {"count": 1}, "prng": [0, 1], "batch_counter": 2}
    written = write_checkpoint_transaction(
        tmp_path / "target",
        run_spec=spec,
        phase_program=contract.phase_program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="toy",
            phase="train_batch",
            program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=slots,
        completed_training_batches=2,
        segment_parent_transaction_id="tx-source",
        segment_start_batch=1,
        segment_batch_count=1,
    )
    row = LaunchRow("toy", "toy", spec)
    monkeypatch.setattr(
        "rlrmp.train.launch.compile_authored_training_intent", lambda _launch: (row,)
    )
    monkeypatch.setattr(
        "rlrmp.train.launch._prepare_execution",
        lambda *_args, **_kwargs: _PreparedExecution(slots, {}, object(), None),
    )
    evidence = verify_resume_authored_training_intent(SimpleNamespace())
    assert evidence[0]["transaction_id"] == written.manifest.transaction_id

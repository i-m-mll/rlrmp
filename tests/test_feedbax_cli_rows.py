"""Regression coverage for governed Feedbax CLI row commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import feedbax
import pytest
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training import ExecutionPreparationRequest

from rlrmp.paths import REPO_ROOT
from rlrmp.train.cs_nominal_gru import CsNominalGruConfig, _config_namespace, write_run_spec
from rlrmp.train.execution_preparation import prepare_cs_supervised
from rlrmp.train.feedbax_cli_rows import build_feedbax_cli_rows_manifest


def test_stage2_builder_emits_three_supported_cli_commands(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runs_dir = repo_root / "results" / "c6c5997" / "runs"
    deploy_dir = repo_root / "results" / "c6c5997" / "deploy"
    runs_dir.mkdir(parents=True)
    deploy_dir.mkdir(parents=True)

    rows = ("flat_3e-5", "rewarm_3e-4", "rewarm_3e-3")
    for row_id in rows:
        args = _config_namespace(CsNominalGruConfig())
        args.n_train_batches = 2
        args.batch_size = 1
        args.n_replicates = 1
        args.hidden_size = 4
        args.dry_run = True
        args.checkpoint_interval_batches = 1
        args.output_dir = str(repo_root / "_artifacts" / "c6c5997" / "runs" / row_id)
        args.spec_dir = str(runs_dir / row_id)
        payload = write_run_spec(args)["run_spec"]
        (runs_dir / f"{row_id}.json").write_text(json.dumps(payload), encoding="utf-8")

    source = {
        "schema_version": 1,
        "rows": [
            {
                "id": row_id,
                "workdir": "/workspace/rlrmp",
                "run_spec": f"results/c6c5997/runs/{row_id}.json",
            }
            for row_id in rows
        ],
    }
    source_path = deploy_dir / "stage2_cli_rows_source.json"
    output_path = deploy_dir / "stage2_rows_manifest.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    manifest = build_feedbax_cli_rows_manifest(source_path, output_path, repo_root=repo_root)

    assert [row["id"] for row in manifest["rows"]] == list(rows)
    for row in manifest["rows"]:
        command = row["command"]
        assert "python -m feedbax execute-training-run-spec" in command
        assert "--resume" in command
        assert "--checkpoint-root" in command
        assert "_run_full_training_from_context" not in command
        nested = deploy_dir / "feedbax_training_run_specs" / f"{row['id']}.json"
        assert json.loads(nested.read_text())["method_ref"]["name"] == "cs_supervised"


def test_real_feedbax_cli_reaches_rlrmp_preparation_before_resume_lookup(tmp_path: Path) -> None:
    """The supported subprocess path builds non-JSON slots before executor resume."""
    args = _config_namespace(CsNominalGruConfig())
    args.n_train_batches = 2
    args.batch_size = 1
    args.n_replicates = 1
    args.hidden_size = 4
    args.dry_run = True
    args.checkpoint_interval_batches = 1
    args.output_dir = str(tmp_path / "artifacts")
    args.spec_dir = str(tmp_path / "spec")
    outer = write_run_spec(args)["run_spec"]
    nested_spec = tmp_path / "training-run-spec.json"
    nested_spec.write_text(
        json.dumps(outer["feedbax_training_run_spec"]),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        "-m",
        "feedbax",
        "execute-training-run-spec",
        str(nested_spec),
        "--checkpoint-root",
        str(tmp_path / "missing-checkpoint"),
        "--resume",
        "--no-progress",
    ]
    feedbax_root = Path(feedbax.__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(feedbax_root))),
    }
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert "_run_full_training_from_context" not in " ".join(command)
    assert completed.returncode != 0
    assert "execution preparation failed" not in completed.stderr.lower()
    assert "checkpoint" in completed.stderr.lower()


def test_legacy_cs_spec_validates_then_preparation_fails_clearly(tmp_path: Path) -> None:
    args = _config_namespace(CsNominalGruConfig())
    args.n_train_batches = 1
    args.batch_size = 1
    args.n_replicates = 1
    args.hidden_size = 4
    args.dry_run = True
    args.output_dir = str(tmp_path / "artifacts")
    args.spec_dir = str(tmp_path / "spec")
    payload = write_run_spec(args)["run_spec"]["feedbax_training_run_spec"]
    payload["method_payload"]["payload"].pop("config")

    legacy_spec = TrainingRunSpec.model_validate(payload)

    with pytest.raises(ValueError, match="predates governed runtime config"):
        prepare_cs_supervised(ExecutionPreparationRequest(run_spec=legacy_spec))

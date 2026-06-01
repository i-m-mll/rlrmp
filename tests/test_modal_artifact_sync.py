from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from rlrmp.modal_artifact_sync import (
    ModalArtifactSyncError,
    build_modal_run_sync_plan,
    sync_modal_run_artifacts,
)
from rlrmp.modal_runner import MODAL_VOLUME_NAME


def test_build_plan_uses_role_based_repo_layout(tmp_path: Path) -> None:
    plan = build_modal_run_sync_plan(
        issue="30f2313",
        run="cs_stochastic_gru__no_hidden_penalty",
        repo_root=tmp_path,
    )

    assert plan.volume_name == MODAL_VOLUME_NAME
    assert plan.remote_spec_dir == "results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty"
    assert (
        plan.remote_artifact_dir == "_artifacts/30f2313/runs/cs_stochastic_gru__no_hidden_penalty"
    )
    assert (
        plan.local_spec_dir
        == tmp_path / "results" / "30f2313" / "runs" / "cs_stochastic_gru__no_hidden_penalty"
    )
    assert (
        plan.local_artifact_dir
        == tmp_path / "_artifacts" / "30f2313" / "runs" / "cs_stochastic_gru__no_hidden_penalty"
    )
    assert plan.spec_command == [
        "modal",
        "volume",
        "get",
        MODAL_VOLUME_NAME,
        "results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty",
        str(plan.local_spec_dir),
    ]


def test_sync_pulls_specs_artifacts_and_validates_run_spec(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    validated: list[Path] = []

    def runner(command: Sequence[str]) -> int:
        commands.append(list(command))
        destination = Path(command[5])
        if command[4].startswith("results/"):
            _write_complete_specs(destination)
        else:
            _write_complete_artifacts(destination, "cs_stochastic_gru__no_hidden_penalty")
        return 0

    results = sync_modal_run_artifacts(
        issue="30f2313",
        runs=["cs_stochastic_gru__no_hidden_penalty"],
        repo_root=tmp_path,
        runner=runner,
        run_spec_validator=validated.append,
    )

    assert len(results) == 1
    assert results[0].validated is True
    assert len(commands) == 2
    assert commands[0][4] == "results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty"
    assert commands[1][4] == "_artifacts/30f2313/runs/cs_stochastic_gru__no_hidden_penalty"
    assert validated == [
        tmp_path
        / "results"
        / "30f2313"
        / "runs"
        / "cs_stochastic_gru__no_hidden_penalty"
        / "run.json"
    ]


def test_sync_accepts_multiple_runs_in_order(tmp_path: Path) -> None:
    remote_paths: list[str] = []

    def runner(command: Sequence[str]) -> int:
        remote_paths.append(command[4])
        destination = Path(command[5])
        run = remote_paths[-1].split("/")[-1]
        if command[4].startswith("results/"):
            _write_complete_specs(destination)
        else:
            _write_complete_artifacts(destination, run)
        return 0

    sync_modal_run_artifacts(
        issue="30f2313",
        runs=["first", "second"],
        repo_root=tmp_path,
        runner=runner,
        run_spec_validator=lambda path: None,
    )

    assert remote_paths == [
        "results/30f2313/runs/first",
        "_artifacts/30f2313/runs/first",
        "results/30f2313/runs/second",
        "_artifacts/30f2313/runs/second",
    ]


def test_sync_rejects_missing_graph_manifest(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> int:
        destination = Path(command[5])
        if command[4].startswith("results/"):
            destination.mkdir(parents=True)
            (destination / "run.json").write_text("{}", encoding="utf-8")
            (destination / "model.graph.json").write_text("{}", encoding="utf-8")
        else:
            _write_complete_artifacts(destination, "missing_manifest")
        return 0

    with pytest.raises(ModalArtifactSyncError, match="model.graph.manifest.json"):
        sync_modal_run_artifacts(
            issue="30f2313",
            runs=["missing_manifest"],
            repo_root=tmp_path,
            runner=runner,
            run_spec_validator=lambda path: None,
        )


def test_sync_rejects_missing_bulk_artifact(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> int:
        destination = Path(command[5])
        if command[4].startswith("results/"):
            _write_complete_specs(destination)
        else:
            destination.mkdir(parents=True)
            (destination / "trained_model.eqx").write_text("", encoding="utf-8")
            (destination / "training_history.eqx").write_text("", encoding="utf-8")
            (destination / "modal_environment.json").write_text("{}", encoding="utf-8")
        return 0

    with pytest.raises(ModalArtifactSyncError, match="training_summary.json"):
        sync_modal_run_artifacts(
            issue="30f2313",
            runs=["missing_artifact"],
            repo_root=tmp_path,
            runner=runner,
            run_spec_validator=lambda path: None,
        )


def test_dry_run_builds_commands_without_running_or_validating(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> int:
        raise AssertionError(f"runner should not be called for dry-run: {command}")

    def validator(path: Path) -> None:
        raise AssertionError(f"validator should not be called for dry-run: {path}")

    results = sync_modal_run_artifacts(
        issue="30f2313",
        runs=["dry_run"],
        repo_root=tmp_path,
        dry_run=True,
        runner=runner,
        run_spec_validator=validator,
    )

    assert results[0].validated is False
    assert results[0].commands[0][4] == "results/30f2313/runs/dry_run"
    assert results[0].commands[1][4] == "_artifacts/30f2313/runs/dry_run"


def test_nonzero_modal_command_aborts_sync(tmp_path: Path) -> None:
    def runner(command: Sequence[str]) -> int:
        return 17

    with pytest.raises(ModalArtifactSyncError, match="exit code 17"):
        sync_modal_run_artifacts(
            issue="30f2313",
            runs=["bad_remote"],
            repo_root=tmp_path,
            runner=runner,
            run_spec_validator=lambda path: None,
        )


def _write_complete_specs(destination: Path) -> None:
    destination.mkdir(parents=True)
    (destination / "run.json").write_text("{}", encoding="utf-8")
    (destination / "model.graph.json").write_text("{}", encoding="utf-8")
    (destination / "model.graph.manifest.json").write_text("{}", encoding="utf-8")


def _write_complete_artifacts(destination: Path, run: str) -> None:
    nested = destination / run
    nested.mkdir(parents=True)
    (nested / "trained_model.eqx").write_text("", encoding="utf-8")
    (nested / "training_history.eqx").write_text("", encoding="utf-8")
    (nested / "training_summary.json").write_text("{}", encoding="utf-8")
    (destination / "modal_environment.json").write_text("{}", encoding="utf-8")

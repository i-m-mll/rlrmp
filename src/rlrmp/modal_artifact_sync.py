"""Sync and validate Modal Volume artifacts for nominal GRU runs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rlrmp.modal_runner import MODAL_VOLUME_NAME, shell_join
from rlrmp.paths import REPO_ROOT
from rlrmp.run_specs import validate_nominal_gru_run_spec_file

Runner = Callable[[Sequence[str]], int]
RunSpecValidator = Callable[[Path], None]

REQUIRED_SPEC_FILES = ("run.json", "model.graph.json", "model.graph.manifest.json")
REQUIRED_ARTIFACT_FILES = (
    "trained_model.eqx",
    "training_history.eqx",
    "training_summary.json",
    "modal_environment.json",
)


class ModalArtifactSyncError(RuntimeError):
    """Raised when Modal artifact sync or local validation fails."""


@dataclass(frozen=True)
class ModalRunSyncPlan:
    """Concrete Modal Volume pull plan for one run."""

    issue: str
    run: str
    volume_name: str
    remote_spec_dir: str
    remote_artifact_dir: str
    local_spec_dir: Path
    local_artifact_dir: Path

    @property
    def spec_command(self) -> list[str]:
        return build_modal_volume_get_command(
            volume_name=self.volume_name,
            remote_path=self.remote_spec_dir,
            local_path=self.local_spec_dir,
        )

    @property
    def artifact_command(self) -> list[str]:
        return build_modal_volume_get_command(
            volume_name=self.volume_name,
            remote_path=self.remote_artifact_dir,
            local_path=self.local_artifact_dir,
        )

    @property
    def commands(self) -> tuple[list[str], list[str]]:
        return (self.spec_command, self.artifact_command)


@dataclass(frozen=True)
class ModalRunSyncResult:
    """Sync outcome for one run."""

    issue: str
    run: str
    local_spec_dir: Path
    local_artifact_dir: Path
    commands: tuple[list[str], list[str]]
    validated: bool
    dry_run: bool


def build_modal_volume_get_command(
    *,
    volume_name: str,
    remote_path: str,
    local_path: Path,
) -> list[str]:
    """Build a ``modal volume get`` command."""

    return ["modal", "volume", "get", volume_name, remote_path, str(local_path)]


def build_modal_run_sync_plan(
    *,
    issue: str,
    run: str,
    repo_root: Path = REPO_ROOT,
    volume_name: str = MODAL_VOLUME_NAME,
) -> ModalRunSyncPlan:
    """Build the Modal Volume pull plan for one run.

    The path layout mirrors the role-based repository convention: tracked
    specs/sidecars live under ``results/`` and bulk outputs live under
    ``_artifacts/``.
    """

    return ModalRunSyncPlan(
        issue=issue,
        run=run,
        volume_name=volume_name,
        remote_spec_dir=f"results/{issue}/runs/{run}",
        remote_artifact_dir=f"_artifacts/{issue}/runs/{run}",
        local_spec_dir=repo_root / "results" / issue / "runs" / run,
        local_artifact_dir=repo_root / "_artifacts" / issue / "runs" / run,
    )


def sync_modal_run_artifacts(
    *,
    issue: str,
    runs: Sequence[str],
    repo_root: Path = REPO_ROOT,
    volume_name: str = MODAL_VOLUME_NAME,
    dry_run: bool = False,
    runner: Runner | None = None,
    run_spec_validator: RunSpecValidator = validate_nominal_gru_run_spec_file,
) -> list[ModalRunSyncResult]:
    """Pull Modal Volume artifacts for runs and validate local completeness."""

    if not runs:
        raise ModalArtifactSyncError("at least one run must be provided")
    command_runner = runner or _run_subprocess
    results: list[ModalRunSyncResult] = []
    for run in runs:
        plan = build_modal_run_sync_plan(
            issue=issue,
            run=run,
            repo_root=repo_root,
            volume_name=volume_name,
        )
        if not dry_run:
            plan.local_spec_dir.parent.mkdir(parents=True, exist_ok=True)
            plan.local_artifact_dir.parent.mkdir(parents=True, exist_ok=True)
            _run_checked(plan.spec_command, runner=command_runner)
            _run_checked(plan.artifact_command, runner=command_runner)
            validate_synced_modal_run(
                plan,
                run_spec_validator=run_spec_validator,
            )
        results.append(
            ModalRunSyncResult(
                issue=issue,
                run=run,
                local_spec_dir=plan.local_spec_dir,
                local_artifact_dir=plan.local_artifact_dir,
                commands=plan.commands,
                validated=not dry_run,
                dry_run=dry_run,
            )
        )
    return results


def validate_synced_modal_run(
    plan: ModalRunSyncPlan,
    *,
    run_spec_validator: RunSpecValidator = validate_nominal_gru_run_spec_file,
) -> None:
    """Validate that a synced run has complete specs, sidecars, and artifacts."""

    missing_specs = [
        file_name
        for file_name in REQUIRED_SPEC_FILES
        if not (plan.local_spec_dir / file_name).is_file()
    ]
    if missing_specs:
        raise ModalArtifactSyncError(
            f"synced run {plan.issue}/{plan.run} is missing tracked spec files: "
            + ", ".join(missing_specs)
        )
    run_spec_validator(plan.local_spec_dir / "run.json")

    if not plan.local_artifact_dir.is_dir():
        raise ModalArtifactSyncError(
            f"synced run {plan.issue}/{plan.run} is missing artifact directory: "
            f"{plan.local_artifact_dir}"
        )
    present_artifact_names = {
        path.name for path in plan.local_artifact_dir.rglob("*") if path.is_file()
    }
    missing_artifacts = [
        file_name
        for file_name in REQUIRED_ARTIFACT_FILES
        if file_name not in present_artifact_names
    ]
    if missing_artifacts:
        raise ModalArtifactSyncError(
            f"synced run {plan.issue}/{plan.run} is missing bulk artifact files: "
            + ", ".join(missing_artifacts)
        )


def _run_checked(command: Sequence[str], *, runner: Runner) -> None:
    returncode = runner(command)
    if returncode != 0:
        raise ModalArtifactSyncError(
            f"Modal Volume command failed with exit code {returncode}: {shell_join(command)}"
        )


def _run_subprocess(command: Sequence[str]) -> int:
    print(shell_join(command), flush=True)
    return subprocess.run(list(command), check=False).returncode


__all__ = [
    "ModalArtifactSyncError",
    "ModalRunSyncPlan",
    "ModalRunSyncResult",
    "REQUIRED_ARTIFACT_FILES",
    "REQUIRED_SPEC_FILES",
    "build_modal_run_sync_plan",
    "build_modal_volume_get_command",
    "sync_modal_run_artifacts",
    "validate_synced_modal_run",
]

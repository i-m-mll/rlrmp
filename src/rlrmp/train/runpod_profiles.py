"""Tracked operational profiles for RLRMP RunPod orchestration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from feedbax.orchestration.drivers.runpod import RunPodDriverConfig


class RunPodOperationalProfile(BaseModel):
    """Non-scientific RunPod transport and environment policy."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "rlrmp.runpod_operational_profile.v1"
    image: str
    gpu_id: str
    datacenters: list[str] = Field(min_length=1)
    local_repos: dict[str, str]
    remote_repos: dict[str, str]
    path_patches: list[tuple[str, str, str]] = Field(default_factory=list)
    overlay_steps: list[str] = Field(default_factory=list)
    auto_teardown: bool = True


def load_runpod_profile(path: Path, *, repo_root: Path) -> RunPodDriverConfig:
    """Load a tracked profile and resolve its explicit repository tokens."""
    profile = RunPodOperationalProfile.model_validate_json(path.read_text(encoding="utf-8"))
    resolved_repo = repo_root.resolve()
    scientific_root = (
        resolved_repo.parents[2]
        if resolved_repo.parent.name == "worktrees"
        else resolved_repo.parent
    )
    tokens = {
        "{repo_root}": resolved_repo,
        "{feedbax_lane0}": scientific_root
        / "20 Feedbax/feedbax/worktrees/feature__e8f90fa-schedule-preflight-discovery",
    }

    def resolve(value: str) -> Path:
        if value in tokens:
            return tokens[value]
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()

    return RunPodDriverConfig(
        image=profile.image,
        gpu_id=profile.gpu_id,
        datacenters=tuple(profile.datacenters),
        local_repos={name: resolve(value) for name, value in profile.local_repos.items()},
        remote_repos=profile.remote_repos,
        path_patches=tuple(profile.path_patches),
        overlay_steps=tuple(profile.overlay_steps),
        auto_teardown=profile.auto_teardown,
    )


def default_runpod_profile_path() -> Path:
    """Return the packaged secure RTX 4090 profile."""
    return Path(__file__).parents[1] / "config" / "runpod_profiles" / "secure_rtx4090.json"


__all__ = [
    "RunPodOperationalProfile",
    "default_runpod_profile_path",
    "load_runpod_profile",
]

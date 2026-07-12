"""Tracked RunPod operational-profile tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rlrmp.train.launch import LaunchRuntimeControls, orchestration_driver_for_bundle

from rlrmp.train.runpod_profiles import (
    default_runpod_profile_path,
    load_runpod_profile,
)


def test_tracked_profile_populates_stock_driver_config(tmp_path: Path) -> None:
    config = load_runpod_profile(default_runpod_profile_path(), repo_root=tmp_path)
    assert config.image.startswith("runpod/pytorch:")
    assert config.gpu_id == "NVIDIA GeForce RTX 4090"
    assert config.datacenters == ("EU-RO-1", "EU-CZ-1", "US-IL-1", "EUR-IS-1")
    assert config.local_repos["rlrmp"] == tmp_path.resolve()
    assert config.local_repos["feedbax"] == Path(
        "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax"
    )
    assert config.remote_repos == {
        "rlrmp": "/workspace/rlrmp",
        "feedbax": "/workspace/feedbax",
    }
    assert config.path_patches == (
        ("/workspace/rlrmp/pyproject.toml", "../20 Feedbax/feedbax", "/workspace/feedbax"),
        ("/workspace/rlrmp/uv.lock", "../20 Feedbax/feedbax", "/workspace/feedbax"),
    )
    assert config.auto_teardown is True


def test_driver_factory_consumes_profile_and_local_needs_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rlrmp.train.orchestration_drivers as drivers

    local = object()
    monkeypatch.setattr(drivers, "local_driver_for_bundle", lambda *args, **kwargs: local)
    assert (
        orchestration_driver_for_bundle(
            SimpleNamespace(),
            driver="local",
            repo_root=tmp_path,
            controls=LaunchRuntimeControls(),
            fork_record=None,
            runpod_profile=None,
        )
        is local
    )
    with pytest.raises(ValueError, match="does not accept"):
        orchestration_driver_for_bundle(
            SimpleNamespace(),
            driver="local",
            repo_root=tmp_path,
            controls=LaunchRuntimeControls(),
            fork_record=None,
            runpod_profile=default_runpod_profile_path(),
        )
    runpod = orchestration_driver_for_bundle(
        SimpleNamespace(),
        driver="runpod",
        repo_root=tmp_path,
        controls=LaunchRuntimeControls(),
        fork_record=None,
        runpod_profile=default_runpod_profile_path(),
    )
    assert runpod.config.gpu_id == "NVIDIA GeForce RTX 4090"

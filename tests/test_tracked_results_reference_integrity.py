"""Guard executable run-spec references into tracked results (issue 2c06960)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from rlrmp.train.distillation_entry import load_distillation_run_spec
from rlrmp.train.distillation_native.closed_loop_kernel import _training_hps_from_spec
from rlrmp.train.training_configs import ClosedLoopDistillationConfig


pytestmark = pytest.mark.feedbax_contract
REPO_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_REFERENCE_KEYS = frozenset({"run_spec", "base_run_spec"})


def _path_references(value: object) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in EXECUTABLE_REFERENCE_KEYS and isinstance(child, str):
                yield str(key), child
            yield from _path_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_references(child)


def _missing_result_references(payload: object, *, repo_root: Path) -> list[str]:
    missing: list[str] = []
    for key, value in _path_references(payload):
        path = value.removeprefix("repo://")
        if path.startswith("results/") and not (repo_root / path).is_file():
            missing.append(f"{key}={value}")
    return missing


def test_tracked_run_specs_have_no_dangling_executable_result_references() -> None:
    failures: list[str] = []
    for path in sorted((REPO_ROOT / "results").glob("*/runs/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for missing in _missing_result_references(payload, repo_root=REPO_ROOT):
            failures.append(f"{path.relative_to(REPO_ROOT)}: {missing}")
    assert failures == []


def test_missing_result_reference_negative_canary() -> None:
    payload = {"base_contract": {"run_spec": "results/deleted/runs/base.json"}}
    assert _missing_result_references(payload, repo_root=REPO_ROOT) == [
        "run_spec=results/deleted/runs/base.json"
    ]


def test_real_closed_loop_spec_loads_from_runtime_owned_base_hps() -> None:
    path = Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json")
    payload = load_distillation_run_spec(
        ClosedLoopDistillationConfig(run_spec=path), method="closed_loop_distillation"
    )
    hps = _training_hps_from_spec(payload)
    assert hps.batch_size == 64
    assert hps.model.hidden_size == 180
    assert payload["base_contract"]["run_spec"] == (
        "results/a378b34/runtime/h0_no_pgd_base_hps.json"
    )
    base = json.loads((REPO_ROOT / payload["base_contract"]["run_spec"]).read_text())
    canonical_hps = json.dumps(base["hps"], sort_keys=True, separators=(",", ":")) + "\n"
    assert hashlib.sha256(canonical_hps.encode()).hexdigest() == base["source"]["hps_sha256"]

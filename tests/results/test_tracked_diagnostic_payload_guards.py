"""Regression guards for tracked diagnostic JSON payload size contracts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
STEADY_STATE_SUMMARY = (
    REPO_ROOT / "results/87424a4/notes/steady_state_perturbation_bank_summary.json"
)
PERTURBATION_RESPONSE_MANIFEST = (
    REPO_ROOT / "tests/fixtures/legacy_payloads/active_perturbation_response_manifest.json"
)
HISTORICAL_PERTURBATION_RESPONSE_MANIFEST = (
    REPO_ROOT / "tests/fixtures/legacy_payloads/historical_perturbation_response_manifest.json"
)
OBJECTIVE_COMPARATOR_SIDECAR = (
    REPO_ROOT / "tests/fixtures/legacy_payloads/objective_comparator_sidecar.json"
)

SCALAR_PROFILE_KEYS = frozenset({"objective_profile"})
RUN_DETAIL_PAYLOAD_KEYS = frozenset(
    {
        "perturbations",
        "raw_rollout_payloads",
        "robust_response_summary",
    }
)
MAX_SCALAR_VALUES = 16
MAX_TRACKED_RESULTS_JSON_BYTES = 500 * 1024
FULL_RUN_SPEC_BYTES = 735 * 1024
FULL_RUN_SPEC_REASON = (
    "Full Feedbax TrainingRunSpec retained for spec-first launch, resume, and fork "
    "provenance; tracked instances use the repository's compact machine-JSON encoding."
)
HISTORICAL_SIZE_EXCEPTIONS: dict[Path, tuple[int, str]] = {
    Path("results/40e1911/notes/perturbation_response_npz_deletion_manifest.json"): (
        600 * 1024,
        "Historical raw-rollout deletion inventory; the payload is a durable list "
        "of deleted paths, not duplicated diagnostic provenance."
    ),
    Path("results/d6d25d6/notes/phase_modulated_recurrent_manifest.json"): (
        700 * 1024,
        "Historical phase-modulated recurrent certificate matrix; the compact "
        "payload is dominated by row-level certificate components rather than "
        "duplicated provenance or bulk arrays.",
    ),
    Path("results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"): (
        FULL_RUN_SPEC_BYTES,
        FULL_RUN_SPEC_REASON,
    ),
    Path("results/c6c5997/runs/rewarm_3e-3-epsilon-ramp.json"): (
        FULL_RUN_SPEC_BYTES,
        FULL_RUN_SPEC_REASON,
    ),
    Path("results/c6c5997/runs/rewarm_3e-4-epsilon-ramp.json"): (
        FULL_RUN_SPEC_BYTES,
        FULL_RUN_SPEC_REASON,
    ),
}


def test_steady_state_summary_is_slim_and_points_to_bulk_detail() -> None:
    payload = _load_json(STEADY_STATE_SUMMARY)

    detail_path = _require_bulk_detail_pointer(payload)
    assert detail_path == "_artifacts/87424a4/notes/steady_state_perturbation_bank_detail.json"
    assert payload["outputs"]["detail_json"] == detail_path

    failures = _dense_payload_failures(
        payload,
        forbid_adapter_payloads=True,
    )
    assert failures == []


def test_active_perturbation_response_manifest_keeps_run_detail_in_bulk_artifact() -> None:
    payload = _load_json(PERTURBATION_RESPONSE_MANIFEST)

    detail_path = _require_bulk_detail_pointer(payload)
    assert detail_path.startswith("_artifacts/020a65b/perturbation_response/")

    failures = _dense_payload_failures(payload)
    assert failures == []

    for run_id, run_payload in payload["runs"].items():
        assert "perturbations" not in run_payload, run_id
        assert "raw_rollout_payloads" not in run_payload, run_id
        assert "robust_response_summary" not in run_payload, run_id
        assert run_payload["perturbation_rows_detail_manifest"] == detail_path
        assert run_payload["robust_response_summary_detail_manifest"] == detail_path


def test_historical_perturbation_response_manifest_keeps_run_detail_in_bulk_artifact() -> None:
    payload = _load_json(HISTORICAL_PERTURBATION_RESPONSE_MANIFEST)

    detail_path = _require_bulk_detail_pointer(payload)
    assert detail_path == (
        "_artifacts/b8aa38e/perturbation_response/"
        "gru_perturbation_response_overnight_robust_proprio_validation_selected_corrected_detail.json"
    )

    failures = _dense_payload_failures(payload)
    assert failures == []

    for run_id, run_payload in payload["runs"].items():
        assert "perturbations" not in run_payload, run_id
        assert "raw_rollout_payloads" not in run_payload, run_id
        assert "robust_response_summary" not in run_payload, run_id
        assert run_payload["perturbation_rows_detail_manifest"] == detail_path
        assert run_payload["robust_response_summary_detail_manifest"] == detail_path
        assert run_payload["raw_rollout_payloads_detail_manifest"] == detail_path


def test_all_tracked_perturbation_response_manifests_avoid_inline_run_detail() -> None:
    manifest_paths = sorted(
        (REPO_ROOT / "results").glob("*/notes/gru_perturbation_response*manifest.json")
    )
    assert manifest_paths

    failures: list[str] = []
    for manifest_path in manifest_paths:
        payload = _load_json(manifest_path)
        if "bank" in payload:
            failures.append(f"{manifest_path.relative_to(REPO_ROOT)}:bank")
        if "bank_summary" in payload:
            summary = payload["bank_summary"]
            if not isinstance(summary, Mapping) or "detail_manifest" not in summary:
                failures.append(f"{manifest_path.relative_to(REPO_ROOT)}:bank_summary")
        runs = payload.get("runs")
        if not isinstance(runs, Mapping):
            continue
        for run_id, run_payload in runs.items():
            if not isinstance(run_payload, Mapping):
                continue
            if "bulk_files" in run_payload:
                failures.append(f"{manifest_path.relative_to(REPO_ROOT)}:{run_id}.bulk_files")
            for key in RUN_DETAIL_PAYLOAD_KEYS:
                if key in run_payload:
                    failures.append(f"{manifest_path.relative_to(REPO_ROOT)}:{run_id}.{key}")

    assert failures == []


def test_tracked_results_json_files_stay_within_size_budget() -> None:
    paths = _tracked_results_json_paths()
    assert not any(
        path.parts[:2] == ("results", "3cd018b") for path in HISTORICAL_SIZE_EXCEPTIONS
    )

    failures: list[str] = []
    for path in paths:
        relpath = path.relative_to(REPO_ROOT)
        size = path.stat().st_size
        if relpath in HISTORICAL_SIZE_EXCEPTIONS:
            max_exception_bytes, reason = HISTORICAL_SIZE_EXCEPTIONS[relpath]
            if size > max_exception_bytes:
                failures.append(
                    f"{relpath} is {size} bytes, above historical exception budget "
                    f"{max_exception_bytes}: {reason}"
                )
            continue
        if size > MAX_TRACKED_RESULTS_JSON_BYTES:
            failures.append(
                f"{relpath} is {size} bytes, above tracked JSON budget "
                f"{MAX_TRACKED_RESULTS_JSON_BYTES}"
            )

    assert failures == []


def test_objective_comparator_sidecar_has_no_dense_values_arrays() -> None:
    payload = _load_json(OBJECTIVE_COMPARATOR_SIDECAR)

    failures = _dense_payload_failures(payload)
    assert failures == []


def test_dense_payload_guard_flags_known_regression_shapes() -> None:
    payload = {
        "bulk_detail_manifest": {"path": "_artifacts/example/detail.json"},
        "runs": {
            "run_a": {
                "perturbations": [{"row_id": "x", "values": list(range(64))}],
                "robust_response_summary": {"raw": [1.0, 2.0]},
            }
        },
        "comparisons": {
            "cmp": {
                "adapter": {"dense": "adapter provenance belongs in bulk detail"},
                "aligned_output_window_profile": [0.0, 1.0, 0.5],
            }
        },
    }

    failures = _dense_payload_failures(payload, forbid_adapter_payloads=True)

    assert any("runs.run_a.perturbations" in failure for failure in failures)
    assert any("runs.run_a.robust_response_summary" in failure for failure in failures)
    assert any("runs.run_a.perturbations[0].values" in failure for failure in failures)
    assert any("comparisons.cmp.adapter" in failure for failure in failures)
    assert any("comparisons.cmp.aligned_output_window_profile" in failure for failure in failures)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    assert isinstance(payload, dict), path
    return payload


def _tracked_results_json_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "results"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(
        REPO_ROOT / relpath.decode("utf-8")
        for relpath in result.stdout.split(b"\0")
        if relpath and relpath.endswith(b".json")
    )


def _require_bulk_detail_pointer(payload: Mapping[str, Any]) -> str:
    detail = payload.get("bulk_detail_manifest")
    assert isinstance(detail, dict)
    path = detail.get("path")
    assert isinstance(path, str)
    assert path.startswith("_artifacts/")
    assert path.endswith(".json")
    return path


def _dense_payload_failures(
    payload: object,
    *,
    forbid_adapter_payloads: bool = False,
) -> list[str]:
    failures: list[str] = []
    for path, value in _walk_json(payload):
        if not path:
            continue
        key = path[-1]
        if not isinstance(key, str):
            continue

        if key in RUN_DETAIL_PAYLOAD_KEYS and (
            _is_under_key(path, "runs") or key != "perturbations"
        ):
            failures.append(f"{_format_path(path)} uses dense run-detail key {key!r}")
            continue

        if _is_dense_profile_key(key):
            failures.append(f"{_format_path(path)} carries a dense profile payload")
            continue

        if forbid_adapter_payloads and "adapter" in key:
            failures.append(f"{_format_path(path)} carries adapter detail in tracked JSON")
            continue

        if key == "values" and _numeric_leaf_count(value) > MAX_SCALAR_VALUES:
            failures.append(
                f"{_format_path(path)} carries {_numeric_leaf_count(value)} numeric values"
            )
    return failures


def _walk_json(
    value: object,
    path: tuple[str | int, ...] = (),
) -> Iterator[tuple[tuple[str | int, ...], object]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_json(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, (*path, index))


def _is_under_key(path: tuple[str | int, ...], key: str) -> bool:
    return key in path[:-1]


def _is_dense_profile_key(key: str) -> bool:
    if key in SCALAR_PROFILE_KEYS:
        return False
    return key in {"profile", "profiles"} or key.endswith("_profile") or "_profile_" in key


def _numeric_leaf_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return 1
    if isinstance(value, list):
        return sum(_numeric_leaf_count(child) for child in value)
    if isinstance(value, Mapping):
        return sum(_numeric_leaf_count(child) for child in value.values())
    return 0


def _format_path(path: tuple[str | int, ...]) -> str:
    chunks: list[str] = []
    for item in path:
        if isinstance(item, int):
            chunks[-1] = f"{chunks[-1]}[{item}]"
        else:
            chunks.append(item)
    return ".".join(chunks)

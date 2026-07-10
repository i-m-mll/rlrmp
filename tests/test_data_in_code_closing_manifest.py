"""Closure contract for the issue 6a298a1 data-in-code baseline drain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "results/6a298a1/notes/closing_manifest.json"
SOURCE_COMMIT = "470ffe0928712fcdbc7cbaf8f3042b5e919f8008"


def _source_baseline() -> tuple[list[str], str]:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:ci/data_in_code_baseline.json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout), hashlib.sha256(result.stdout).hexdigest()


def test_closing_manifest_resolves_every_starting_key_once() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    starting, source_sha256 = _source_baseline()

    assert manifest["schema_id"] == "rlrmp.data_in_code_closing_manifest"
    assert manifest["schema_version"] == "rlrmp.data_in_code_closing_manifest.v1"
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["source_baseline_sha256"] == source_sha256
    assert manifest["starting_entry_count"] == 136 == len(starting)
    assert set(manifest["resolutions"]) == set(starting)

    for key, resolution in manifest["resolutions"].items():
        assert resolution["route"]
        assert resolution["destination"]
        assert len(resolution["rationale"].strip()) >= 40, key


def test_closing_manifest_records_ten_or_more_load_proofs() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    proofs = [
        (key, resolution["load_proof"])
        for key, resolution in manifest["resolutions"].items()
        if "load_proof" in resolution
    ]

    assert len(proofs) >= 10
    for key, proof in proofs:
        assert proof["status"] == "passed", key
        assert proof["test"], key
